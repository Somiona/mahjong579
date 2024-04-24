import asyncio
import logging
import os
import random
import sys
import threading
import time
import traceback
from collections import OrderedDict, defaultdict
from queue import Queue

import numpy as np
import torch
from quart.utils import run_sync

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from typing import List

from mahjong.agent import AiAgent
from mahjong.check_agari import check_riichi
from mahjong.display import (
    TENHOU_TILE_STRING_DICT,
    TILE_STRING_DICT,
    blue,
    cyan,
    green,
    red,
    yellow,
)
from mahjong.game import MahjongGame
from mahjong.yaku import Yaku, YakuList
from model.models import RewardPredictor


class ControlledQueue(Queue):
    def __init__(self, maxsize=0):
        super(ControlledQueue, self).__init__(maxsize)
        self._allow_put = True

    def allow_put(self):
        self._allow_put = True

    def put(self, item, block: bool = ..., timeout=...) -> None:
        if self._allow_put:
            self._allow_put = False
            super().put(item, block, timeout)


class Client(object):
    def __init__(self, username):
        self.username = username
        self.message_queue = ControlledQueue()

    def __eq__(self, username):
        return self.username == username

    def recv(self):
        buffer = []
        while True:
            data = self.client_socket.recv(1)
            if len(data) == 0:
                break
            if data == b"\n":
                break
            buffer.append(data)
        return b"".join(buffer).decode("utf-8")

    def fetch_message(self):
        self.message_queue.allow_put()
        return self.message_queue.get()


class GameEnvironment(object):
    def __init__(self, has_aka=True, min_score=0, fast=False, train=False):
        self.game = MahjongGame(has_aka, is_playback=False)
        self.agents = self.game.agents
        self.round = 0
        self.honba = 0
        self.riichi_ba = 0
        self.has_aka = has_aka

        self.clients = []
        self.observe_info = defaultdict(list)  # {who: [observer_client]}
        self.observers = {}  # {username: (observe_who, observer_client)}

        self.current_player = 0
        self.game_start = False
        self.AI_count = 4
        self.ai_agent = AiAgent()
        self.min_score = min_score
        self.fast = fast
        self.train = train
        if train:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            params = torch.load(
                "output/reward-model/checkpoints/best.pt", map_location=self.device
            )
            logging.debug(green("Reward model loaded"))
            self.reward_model = RewardPredictor(
                74, params["hidden_dims"], params["num_layers"]
            )
            self.reward_model.load_state_dict(params["state_dict"])
            self.reward_model.to(self.device)
            self.collected_data = defaultdict(list)
            self.reward_features = defaultdict(list)
        for i in range(self.AI_count):
            self.clients.append(Client(f"一姬{i + 1}(简单)"))

    def reward(self, features, i):
        """计算第i轮的reward"""
        features = features.to(self.device)
        if i == 0:
            score0 = 250
            score1 = self.reward_model(features[:, :1, :]).item() * 500
        else:
            score0 = self.reward_model(features[:, :i, :]).item() * 500
            score1 = self.reward_model(features[:, : i + 1, :]).item() * 500
        return score1 - score0

    def start(self):
        self.game.new_game(self.round, self.honba, self.riichi_ba)

    def reset(self):
        logging.info("Game is reset")
        self.game = MahjongGame(self.has_aka, is_playback=False)
        self.agents = self.game.agents
        self.round = 0
        self.honba = 0
        self.riichi_ba = 0

        self.clients.clear()
        self.observe_info.clear()
        self.observers.clear()

        self.current_player = 0
        self.game_start = False
        if self.train:
            # self.collected_data.clear()
            self.reward_features.clear()
        for i in range(self.AI_count):
            self.clients.append(Client(f"一姬{i + 1}(简单)"))

    def fetch_decision_message(self, client: Client, actions, after_tsumo):
        who = self.clients.index(client)
        return self.decision_by_ai(who, actions, after_tsumo)

    def fetch_discard_message(self, who, client: Client, tiles, banned):
        if tiles == "all":
            tiles = list(self.agents[who].tiles)
        return self.discard_by_ai(who, tiles, banned)

    def decision_by_ai(self, who, actions, after_tsumo):
        state = self.game.get_feature(who)
        pon_action = None
        chi_actions = {}
        kan_actions = []
        pon_feature = None
        action_score_dict = {}

        for i, action in enumerate(actions):
            if action["type"] == "agari":
                if self.ai_agent.agari_decision(self.agents, action):
                    action_score_dict[i] = 1
            elif action["type"] == "riichi":
                score = self.ai_agent.riichi_decision(state)
                logging.debug(
                    yellow(
                        f"「{self.clients[who].username}」「立直」行为意愿: {score:.3f}"
                    )
                )
                action_score_dict[i] = score
            elif action["type"] == "ryuukyoku":
                if action["kyuuhai_type_count"] == 9:
                    if (
                        self.agents[who].score >= 100
                    ):  # 只有9种9牌且分数没那么低时还是流了吧
                        action_score_dict[i] = 1
            elif action["type"] == "pon":
                pattern = action["pattern"]
                if pon_feature is None:
                    pon_feature = self.game.get_pon_feature(
                        who, pattern[0] // 4, action["kui"]
                    )
                if pon_action is None:
                    pon_action = i, pon_feature
                else:  # 有多种碰的方法，则一定存在赤牌，方便起见，优先考虑将赤牌碰出去的操作
                    if {16, 52, 88}.intersection(pattern):
                        pon_action = i, pon_feature
            elif action["type"] == "chi":
                pattern = action["pattern"]
                chi_ptn = min(pattern) // 4
                chi_feature = chi_actions.get(
                    chi_ptn, (i, self.game.get_chi_feature(who, chi_ptn, action["kui"]))
                )[1]
                if chi_ptn not in chi_actions:
                    chi_actions[chi_ptn] = i, chi_feature
                else:  # 同一种顺子pattern有多种吃的方法，则一定存在赤牌，方便起见，优先考虑将赤牌吃出去的操作
                    if {16, 52, 88}.intersection(pattern):
                        chi_actions[chi_ptn] = i, chi_feature
            elif action["type"] == "kan":
                pattern = action["pattern"]
                kan_feature = self.game.get_kan_feature(who, pattern)
                kan_actions.append((i, kan_feature))
        if pon_action:
            i, pon_feature = pon_action
            pon_state = np.concatenate([pon_feature, state], axis=0)
            action_score_dict[i] = score = self.ai_agent.pon_decision(pon_state)
            logging.debug(
                yellow(f"「{self.clients[who].username}」「碰」行为意愿: {score:.3f}")
            )
        if chi_actions:
            for i, chi_feature in chi_actions.values():
                chi_state = np.concatenate([chi_feature, state], axis=0)
                action_score_dict[i] = score = self.ai_agent.chi_decision(chi_state)
                logging.debug(
                    yellow(
                        f"「{self.clients[who].username}」「吃」行为意愿: {score:.3f}"
                    )
                )
        if kan_actions:
            for i, kan_feature in kan_actions:
                kan_state = np.concatenate([kan_feature, state], axis=0)
                action_score_dict[i] = score = self.ai_agent.kan_decision(kan_state)
                logging.debug(
                    yellow(
                        f"「{self.clients[who].username}」「杠」行为意愿: {score:.3f}"
                    )
                )
        if action_score_dict:
            max_score_action, max_score = max(
                action_score_dict.items(), key=lambda x: x[1]
            )
            if max_score < 0.5:  # 行为意愿均低于阈值，选择pass
                if not after_tsumo and not self.fast and random.random() < 0.7:
                    time.sleep(1 + random.random() * 3)
                return actions[0]
            if not self.fast:
                time.sleep(1 + random.random() * 3)
            return actions[max_score_action]
        return actions[0]

    def discard_by_ai(self, who, tiles, banned):
        if not self.fast:
            time.sleep(1 + random.random() * 2)
        if banned:
            tiles = [_ for _ in tiles if _ // 4 not in banned]
        state = self.game.get_feature(who)
        discard, conf = self.ai_agent.discard(state, tiles)
        logging.debug(
            yellow(
                f"「{self.clients[who].username}」以置信度:{conf:.3f} 切出「{TENHOU_TILE_STRING_DICT[discard]}」"
            )
        )
        if self.train:
            self.collected_data[who].append([state, discard // 4])
        return discard

    def print_agari_info(self, who, from_who, action):
        han = action["han"]
        fu = action["fu"]
        score = action["score"]
        ret = action["yaku"]
        yaku_list = action["yaku_list"]
        if who != from_who:
            agari_info = f"「{self.clients[from_who].username}」放铳！「{self.clients[who].username}」荣和！役种: {'、'.join(yaku_list)}->"
        else:
            agari_info = (
                f"「{self.clients[who].username}」自摸！役种: {'、'.join(yaku_list)}->"
            )
        if isinstance(ret, List):
            if han >= 2:
                agari_info += f"{han}倍役满！"
            else:
                agari_info += "役满！"
        else:
            agari_info += f"{han}番({fu}符)->基本点: {score}"
        logging.info(cyan(agari_info))
        self.agents[who].tiles.difference_update({action["machi"]})
        logging.info(
            cyan(
                self.agents[who].display_tiles()
                + "  "
                + TENHOU_TILE_STRING_DICT[action["machi"]]
            )
        )
        if self.agents[who].furo:
            logging.info(cyan(self.agents[who].display_furo()))

    def game_update(self, res):
        change_oya = True
        self.honba = honba = self.game.honba
        self.riichi_ba = riichi_ba = self.game.riichi_ba
        oya = self.game.oya
        score_delta = [0, 0, 0, 0]
        if isinstance(res, list):  # 和牌
            first_winner = res[0]["who"]
            for action in res:
                who = action["who"]
                from_who = action["from_who"]
                score = action["score"]
                if who == from_who:  # 自摸
                    if who == oya:
                        score = ((score * 2) + 90) // 100
                        for i in range(4):
                            if i != oya:
                                self.agents[i].score -= score + honba
                                score_delta[i] -= score + honba
                        self.agents[who].score += score * 3 + honba * 3
                        score_delta[who] += score * 3 + honba * 3
                        change_oya = False
                    else:
                        score_oya = ((score * 2) + 90) // 100
                        score = (score + 90) // 100
                        for i in range(4):
                            if i == who:
                                self.agents[i].score += (
                                    score_oya + score * 2 + honba * 3
                                )
                                score_delta[i] += score_oya + score * 2 + honba * 3
                            elif i == oya:
                                self.agents[i].score -= score_oya + honba
                                score_delta[i] -= score_oya + honba
                            else:
                                self.agents[i].score -= score + honba
                                score_delta[i] -= score + honba
                else:
                    if who == oya:
                        score = ((score * 6) + 90) // 100 + honba * 3
                        change_oya = False
                    else:
                        score = ((score * 4) + 90) // 100 + honba * 3
                    self.agents[from_who].score -= score
                    self.agents[who].score += score
                    score_delta[from_who] -= score
                    score_delta[who] += score
                self.print_agari_info(who, from_who, action)
            if riichi_ba:
                self.agents[first_winner].score += riichi_ba * 10
                score_delta[first_winner] += riichi_ba * 10
            self.riichi_ba = 0
            if not change_oya:
                self.honba += 1
            else:
                self.honba = 0
        else:  # 流局
            why = res["why"]
            if why == "yama_end":  # 结算荒牌流局
                logging.info(cyan("荒牌流局"))
                nagashimangan = res["nagashimangan"]
                machi_state = res["machi_state"]
                for i in range(4):
                    if i in machi_state:
                        machi_tiles = machi_state[i][1]
                        logging.info(
                            cyan(
                                f"「{self.clients[i].username}」听牌: {'、'.join(TILE_STRING_DICT[_] for _ in machi_tiles)}"
                            )
                        )
                change_oya = oya not in machi_state
                if nagashimangan:  # 流满
                    for i in nagashimangan:
                        logging.info(cyan(f"「{self.clients[i].username}」流局满贯！"))
                        for j in range(4):
                            if j == i:
                                if j == oya:
                                    self.agents[j].score += 120
                                    score_delta[j] += 120
                                else:
                                    self.agents[j].score += 80
                                    score_delta[j] += 80
                            else:
                                if j == oya:
                                    self.agents[j].score -= 40
                                    score_delta[j] -= 40
                                else:
                                    self.agents[j].score -= 20
                                    score_delta[j] -= 20
                else:
                    if 1 <= len(machi_state) < 4:
                        score_get = 30 // len(machi_state)
                        score_give = 30 // (4 - len(machi_state))
                        for i in range(4):
                            if i in machi_state:
                                self.agents[i].score += score_get
                                score_delta[i] += score_get
                            else:
                                self.agents[i].score -= score_give
                                score_delta[i] -= score_give
            else:
                if why == "yao9":
                    who = res["who"]
                    logging.info(
                        cyan(f"流局: 「{self.clients[who].username}」九种九牌")
                    )
                elif why == "kaze4":
                    logging.info(cyan("流局: 四风连打"))
                elif why == "kan4":
                    logging.info(cyan("流局: 四杠散了"))
                elif why == "reach4":
                    logging.info(cyan("流局: 四家立直"))
                elif why == "ron3":
                    logging.info(cyan("流局: 三家和了"))
                change_oya = False
            self.honba += 1
        if change_oya:
            self.round += 1
        if min(p.score for p in self.agents) * 100 < self.min_score:
            return True, score_delta
        if self.round > 11:
            return True, score_delta
        if self.round > 7 or (self.round == 7 and not change_oya):
            if max(p.score for p in self.agents) < 300:
                return False, score_delta
            if change_oya:
                if self.riichi_ba:
                    winner = max(
                        ((i, p.score) for i, p in enumerate(self.agents)),
                        key=lambda x: x[1],
                    )[0]
                    self.agents[winner].score += self.riichi_ba * 10
                    self.riichi_ba = 0
                return True, score_delta
            winner = self.game.get_rank()[0][0]
            return winner == oya, score_delta
        else:
            return False, score_delta

    def get_game_info(self):
        return {
            "round": self.game.round,
            "honba": self.game.honba,
            "riichi_ba": self.game.riichi_ba,
            "dora_indicator": self.game.dora_indicator,
            "oya": self.game.oya,
            "agents": [
                {
                    "username": self.clients[i].username,
                    "score": p.score,
                    "tile_count": len(self.agents[i].tiles),
                    "furo": OrderedDict({str(x): y for x, y in p.furo.items()}),
                    "kui_info": p.kui_info,
                    "riichi": p.riichi_status,
                    "riichi_round": p.riichi_round,
                    "discard": p.discard_tiles,
                    "river": p.river,
                    "riichi_tile": p.riichi_tile,
                    "is_ai": self.clients[i].is_ai(),
                }
                for i, p in enumerate(self.agents)
            ],
            "left_num": self.game.left_num,
        }

    def get_player_info(self, who):
        p = self.agents[who]
        return {
            "username": self.clients[who].username,
            "seat": who,
            "tiles": list(p.tiles),
            "furo": OrderedDict({str(x): y for x, y in p.furo.items()}),
            "kui_info": p.kui_info,
            "machi": list(sorted(p.machi)),
        }

    @run_sync
    def check_draw(self, who, tile_id, where):
        """
        检查并响应当前是否能和牌、立直、暗杠、加杠等，生成一个可行的行为列表并选择

        :return {
            'type': 'agari' / 'riichi' / 'kan' / 'pass' / 'ryuukyoku'
            'who': who,
            'from_who': who,
            'pattern'
        }
        """
        player = self.agents[who]
        connection = self.clients[who]
        actions = [{"type": "pass"}]
        can_agari = False
        """判定和牌"""
        agari = None
        yaku = None
        tenhou = False  # 天、地和
        if tile_id // 4 in player.machi:
            if where == -1:
                tokusyu = 1
            elif self.game.left_num == 0:
                tokusyu = 3
            else:
                tokusyu = 0
            yaku = Yaku(
                hand_tiles=player.tiles,
                furo=player.furo,
                agarihai=tile_id,
                dora=self.game.dora,
                ura_dora=self.game.ura_dora,
                bahai=self.game.round_wind,
                menfon=player.menfon,
                tsumo=True,
                riichi=player.riichi_status,
                ippatsu=player.ippatsu_status,
                tokusyu=tokusyu,
                aka=self.has_aka,
            )
            agari = yaku.agari
            if self.game.first_round:
                tenhou = True
                actions.append(
                    {"type": "agari", "who": who, "from_who": who, "machi": tile_id}
                )
                can_agari = True
            elif yaku.naive_check_yaku():
                actions.append(
                    {"type": "agari", "who": who, "from_who": who, "machi": tile_id}
                )
                can_agari = True
            else:
                if isinstance(agari, str):
                    x = list(map(lambda _: int(_, 16), agari.split(",")))
                    han, fu, score, ret = yaku.yaku(x)
                else:
                    if yaku.counter[yaku.agarihai] == 2:
                        ret = [YakuList.KOKUSHIJUSANMEN]
                        han = 2
                    else:
                        ret = [YakuList.KOKUSHIMUSO]
                        han = 1
                    fu = 25
                    score = han * 8000
                if han > 0:
                    actions.append(
                        {
                            "type": "agari",
                            "who": who,
                            "from_who": who,
                            "yaku": ret,
                            "han": han,
                            "fu": fu,
                            "score": score,
                            "machi": tile_id,
                        }
                    )
                    can_agari = True
        """判定九种九牌"""
        if self.game.first_round:
            kyuuhai_type_count = len(
                {_ // 4 for _ in player.tiles}.intersection(
                    [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
                )
            )
            if kyuuhai_type_count >= 9:
                actions.append(
                    {
                        "type": "ryuukyoku",
                        "who": who,
                        "why": "yao9",
                        "kyuuhai_type_count": kyuuhai_type_count,
                    }
                )
        """判定立直"""
        can_riichi = self.game.can_declare_riichi(who)
        if can_riichi:
            actions.append(
                {
                    "type": "riichi",
                    "who": who,
                    "step": 1,
                    "double_riichi": self.game.first_round,
                }
            )

        """判定杠"""
        can_ankan, ankan_patterns = self.game.check_kan(who, tile_id, mode=0)
        can_addkan, addkan_pattern = self.game.check_kan(who, tile_id, mode=2)
        if can_ankan or can_addkan:
            for ptn in ankan_patterns + addkan_pattern:
                actions.append(
                    {"type": "kan", "pattern": ptn, "who": who, "from_who": who}
                )
        if len(actions) > 1:
            message = {"event": "decision", "actions": actions}
            action = self.decision_by_ai(who, actions, True)
            if action["type"] == "agari":
                if "yaku" not in action:
                    if isinstance(agari, str):
                        x = list(map(lambda _: int(_, 16), agari.split(",")))
                        han, fu, score, ret = yaku.yaku(x)
                    else:
                        if yaku.counter[yaku.agarihai] == 2:
                            ret = [YakuList.KOKUSHIJUSANMEN]
                            han = 2
                        else:
                            ret = [YakuList.KOKUSHIMUSO]
                            han = 1
                        fu = 25
                        score = han * 8000
                    if tenhou:
                        han += 1
                        if isinstance(ret, list):
                            if who == self.game.oya:
                                tenhou = YakuList.TENHOU
                                if YakuList.KOKUSHIMUSO in ret:
                                    ret.remove(YakuList.KOKUSHIMUSO)
                                    ret.append(YakuList.KOKUSHIJUSANMEN)
                                    han += 1
                                elif YakuList.CHURENPOTO in ret:
                                    ret.remove(YakuList.CHURENPOTO)
                                    ret.append(YakuList.CHURENCHUMEN)
                                    han += 1
                                elif YakuList.SUANKO in ret:
                                    ret.remove(YakuList.SUANKO)
                                    ret.append(YakuList.SUANKOTANKI)
                                    han += 1
                            else:
                                tenhou = YakuList.CHIHOU
                            ret.append(tenhou)
                            score += 8000
                        else:
                            ret = [
                                YakuList.TENHOU
                                if who == self.game.oya
                                else YakuList.CHIHOU
                            ]
                            han = 1
                            score = 8000
                    action["yaku"] = ret
                    action["han"] = han
                    action["fu"] = fu
                    action["score"] = score
                action["yaku_list"] = yaku.parse_yaku_ret(action["yaku"], True)
                action["hai"] = yaku.hand_tiles
                action["furo"] = list(yaku.furo.values())
            elif can_agari:  # 见逃
                if player.riichi_status:
                    player.riichi_furiten = True
        else:
            action = None
        return action

    @run_sync
    def check_discard(self, who, from_who, is_next_player, tile_id, add_kan=False):
        """
        检查并响应别家打出的牌是否能和牌、吃、碰、明杠，返回一个可行的行为列表并选择，add_kan=True时只判断抢杠和牌

        :return {
            'type': 'agari' / 'chi' / 'pon' / 'kan' / 'pass'
            'pattern': [int]
        }
        """
        player = self.agents[who]
        connection = self.clients[who]
        actions = [{"type": "pass", "who": who}]
        can_agari = False
        yaku = None
        agari = None
        if not player.furiten:
            """判定和牌"""
            if tile_id // 4 in player.machi:
                if add_kan:
                    tokusyu = 2
                elif self.game.left_num == 0:
                    tokusyu = 3
                else:
                    tokusyu = 0
                yaku = Yaku(
                    hand_tiles=player.tiles.union({tile_id}),
                    furo=player.furo,
                    agarihai=tile_id,
                    dora=self.game.dora,
                    ura_dora=self.game.ura_dora,
                    bahai=self.game.round_wind,
                    menfon=player.menfon,
                    tsumo=False,
                    riichi=player.riichi_status,
                    ippatsu=player.ippatsu_status,
                    tokusyu=tokusyu,
                    aka=self.has_aka,
                )
                agari = yaku.agari
                if yaku.naive_check_yaku():
                    actions.append(
                        {
                            "type": "agari",
                            "who": who,
                            "from_who": from_who,
                            "machi": tile_id,
                        }
                    )
                    can_agari = True
                else:
                    if isinstance(agari, str):
                        x = list(map(lambda _: int(_, 16), agari.split(",")))
                        han, fu, score, ret = yaku.yaku(x)
                    else:
                        if yaku.counter[yaku.agarihai] == 2:
                            ret = [YakuList.KOKUSHIJUSANMEN]
                            han = 2
                        else:
                            ret = [YakuList.KOKUSHIMUSO]
                            han = 1
                        fu = 25
                        score = han * 8000
                    if han > 0:
                        actions.append(
                            {
                                "type": "agari",
                                "who": who,
                                "from_who": from_who,
                                "yaku": ret,
                                "han": han,
                                "fu": fu,
                                "score": score,
                                "machi": tile_id,
                            }
                        )
                        can_agari = True
                    else:
                        player.round_furiten = True
        if not add_kan:
            """判断吃碰杠"""
            if is_next_player:
                can_chi, patterns = self.game.check_chi(who, tile_id)
                for pattern in patterns:
                    for furo in player.search_furo(0, pattern, tile_id):
                        actions.append(
                            {
                                "type": "chi",
                                "who": who,
                                "from_who": from_who,
                                "pattern": furo,
                                "kui": tile_id,
                            }
                        )
            can_pon, pattern = self.game.check_pon(who, tile_id)
            if can_pon:
                for furo in player.search_furo(1, pattern, tile_id):
                    actions.append(
                        {
                            "type": "pon",
                            "who": who,
                            "from_who": from_who,
                            "pattern": furo,
                            "kui": tile_id,
                        }
                    )
            can_kan, pattern = self.game.check_kan(who, tile_id, mode=1)
            if can_kan:
                actions.append(
                    {
                        "type": "kan",
                        "who": who,
                        "from_who": from_who,
                        "pattern": pattern,
                        "kui": tile_id,
                    }
                )
        if len(actions) > 1:
            message = {"event": "decision", "actions": actions}
            action = self.decision_by_ai(who, actions, False)
            if action["type"] == "agari":
                if "yaku" not in action:
                    if isinstance(agari, str):
                        x = list(map(lambda _: int(_, 16), agari.split(",")))
                        han, fu, score, ret = yaku.yaku(x)
                    else:
                        if yaku.counter[yaku.agarihai] == 2:
                            ret = [YakuList.KOKUSHIJUSANMEN]
                            han = 2
                        else:
                            ret = [YakuList.KOKUSHIMUSO]
                            han = 1
                        fu = 25
                        score = han * 8000
                    action["yaku"] = ret
                    action["han"] = han
                    action["fu"] = fu
                    action["score"] = score
                action["yaku_list"] = yaku.parse_yaku_ret(action["yaku"], False)
                action["hai"] = yaku.hand_tiles
                action["furo"] = list(yaku.furo.values())
            elif can_agari:  # 见逃
                if player.riichi_status:
                    player.riichi_furiten = True
                player.round_furiten = True
        else:
            action = None
        return action

    async def handle_draw(self, who, tile_id=None, where=0):
        tile_id = self.game.draw(who=who, tile_id=tile_id, where=where)
        connection = self.clients[who]

        message = {"event": "draw", "who": who, "tile_id": tile_id, "where": where}
        action = await self.check_draw(who, tile_id, where)
        return tile_id, action

    async def random_delay(self):
        if not self.fast and random.random() < 0.1:
            await asyncio.sleep(1 + random.random() * 3)
        return {"type": "pass"}

    async def handle_discard(
        self, who, tile_id, mode, after_tsumo=True, is_riichi_tile=False
    ):
        """
        who: 切牌者
        tile_id: 切出的牌
        mode: 是否为摸切
        after_tsumo: 是否为摸牌以后的切牌（还可能是鸣牌之后的切牌）
        is_riichi_tile: 切出的是否为立直宣言牌
        """
        self.game.discard(who=who, tile_id=tile_id)
        connection = self.clients[who]
        agari_actions = []
        pon_kan_action = None
        chi_action = None
        jobs = []
        for i in range(1, 4):
            player_pos = (who + i) % 4
            jobs.append(self.check_discard(player_pos, who, i == 1, tile_id))
        jobs.append(self.random_delay())
        actions = await asyncio.gather(*jobs)
        for action in actions:
            if isinstance(action, dict):
                if action["type"] == "agari":
                    agari_actions.append(action)
                elif action["type"] in ["pon", "kan"]:
                    pon_kan_action = action
                elif action["type"] == "chi":
                    chi_action = action
        if agari_actions:
            return agari_actions
        if pon_kan_action:
            return pon_kan_action
        if chi_action:
            return chi_action
        if not self.fast:
            await asyncio.sleep(0.4)

    async def handle_tsumo_action(self, tile_id, action):
        """玩家摸牌以后的行为"""
        p = self.game.agents[self.current_player]
        while action is not None:
            if action["type"] == "pass":
                return tile_id, action
            elif action["type"] == "agari":
                """处理和牌"""
                return tile_id, action
            elif action["type"] == "riichi":
                """处理立直宣言"""
                logging.info(
                    blue(f"「{self.clients[self.current_player].username}」「立直」!")
                )
                p.declare_riichi = 1
                return tile_id, action
            elif action["type"] == "ryuukyoku":
                return tile_id, action
            elif action["type"] == "kan":
                """处理杠"""
                self.game.first_round = False  # 清除第一巡标记
                for _ in self.agents:
                    _.ippatsu_status = 0  # 清除所有玩家的一发标识

                kan_type, pattern, add = action["pattern"]
                if kan_type == 0:
                    kan_tile_list = p.search_furo(4, pattern, add)
                    self.game.kan(
                        self.current_player,
                        kan_tile_list,
                        from_who=self.current_player,
                        mode=0,
                    )
                    logging.info(
                        blue(
                            f"「{self.clients[self.current_player].username}」暗杠「{' '.join(TENHOU_TILE_STRING_DICT[_] for _ in kan_tile_list)}」"
                        )
                    )
                else:
                    kan_tile_list = p.search_furo(2, pattern, add)
                    agari_actions = []
                    jobs = []
                    for i in range(1, 4):
                        player_pos = (self.current_player + i) % 4
                        jobs.append(
                            self.check_discard(
                                player_pos,
                                self.current_player,
                                i == 1,
                                add,
                                add_kan=True,
                            )
                        )
                    jobs.append(self.random_delay())
                    actions = await asyncio.gather(*jobs)
                    for act in actions:
                        if act is None:
                            continue
                        if act["type"] == "agari":
                            agari_actions.append(act)
                    if agari_actions:
                        return tile_id, agari_actions
                    self.game.kan(
                        self.current_player,
                        kan_tile_list,
                        from_who=self.current_player,
                        add=add,
                        mode=2,
                    )
                    logging.info(
                        blue(
                            f"「{self.clients[self.current_player].username}」加杠「{' '.join(TENHOU_TILE_STRING_DICT[_] for _ in kan_tile_list)}」"
                        )
                    )
                self.game.new_dora()
                tile_id, action = await self.handle_draw(
                    who=self.current_player, where=-1
                )
        return tile_id, action

    def select_tile(
        self, client, tiles, banned=None, tsumo=None, riichi=False, is_riichi_tile=False
    ):
        if riichi or is_riichi_tile:
            return tsumo, True
        banned = banned or []
        who = self.clients.index(client)

        if tiles == "all":
            tiles = list(self.agents[who].tiles)
        tile_id = self.discard_by_ai(who, tiles, banned)
        mode = (
            tile_id == tsumo
        )  # 是否为摸切。如果banned为空，则tsumo为自摸的牌，否则tsumo为被鸣的牌，则必定不是摸切
        return tile_id, mode

    async def game_loop(self):
        self.current_player = self.game.oya
        p = self.agents[self.current_player]
        connection = self.clients[self.current_player]
        tile_id, action = await self.handle_draw(who=self.current_player, where=0)
        tile_id, res = await self.handle_tsumo_action(tile_id, action)
        if res:  # 自摸和了或者加杠被人抢和
            if isinstance(res, list):
                if len(res) == 3:
                    return {"event": "ryuukyoku", "why": "ron3", "action": res}
                return res
            event_type = res.get("type")
            if event_type == "agari":
                return [res]
            if event_type == "ryuukyoku":
                return res
        banned = []
        after_tsumo = True
        while 1:
            is_riichi_tile = p.declare_riichi and p.riichi_tile == -1
            """玩家选择一张牌"""
            if p.riichi_status:
                tile_id, mode = self.select_tile(
                    connection,
                    [tile_id],
                    tsumo=tile_id,
                    riichi=True,
                    is_riichi_tile=is_riichi_tile,
                )
                #!
            elif p.declare_riichi:
                riichi_options = check_riichi(
                    p.hand_tile_counter, return_riichi_hai=True
                )
                tile_id, mode = self.select_tile(
                    connection,
                    [_ for _ in p.tiles if _ // 4 in riichi_options],
                    tsumo=tile_id,
                    is_riichi_tile=is_riichi_tile,
                )
            else:
                tile_id, mode = self.select_tile(
                    connection,
                    "all",
                    banned=banned,
                    tsumo=tile_id,
                    is_riichi_tile=is_riichi_tile,
                )
            if p.declare_riichi and p.riichi_tile == -1:  # 立直时设置横放牌
                p.riichi_tile = tile_id
            banned.clear()
            actions = await self.handle_discard(
                self.current_player,
                tile_id=tile_id,
                mode=mode,
                after_tsumo=after_tsumo,
                is_riichi_tile=is_riichi_tile,
            )  # pass时, actions=None
            if not self.game_start:
                return None
            if isinstance(actions, list):  # 有人和了
                if len(actions) == 3:
                    return {"event": "ryuukyoku", "why": "ron3", "action": actions}
                return actions

            if (
                self.game.first_round
                and p.menfon == 30
                and 27 <= p.discard_tiles[0] // 4 <= 30
            ):  # 第一巡四风连打判定
                if all(
                    _.discard_tiles
                    and _.discard_tiles[0] // 4 == p.discard_tiles[0] // 4
                    for _ in self.agents
                ):
                    return {"type": "ryuukyoku", "why": "kaze4"}

            if (
                p.declare_riichi and not p.riichi_status
            ):  # 没人和牌并且自己宣告立直的情况下，成功立个直
                self.game.riichi(
                    self.current_player, double_riichi=self.game.first_round
                )
                if all(_.riichi_status for _ in self.agents):
                    return {"type": "ryuukyoku", "why": "reach4"}
            if p.menfon == 30:
                self.game.first_round = False
            if actions is not None:  # 其他玩家的操作
                p.river.pop()
                if tile_id == p.riichi_tile:  # 立直宣言牌被鸣了，将其移除
                    p.riichi_tile = -1
                after_tsumo = False
                p.nagashimangan = 0  # 清除被鸣牌玩家的流局满贯标识
                self.game.first_round = False  # 清除第一巡标识
                for _ in self.agents:
                    _.ippatsu_status = 0  # 清除所有玩家的一发标识
                player_id = actions["who"]
                self.current_player = player_id
                p = self.agents[self.current_player]
                connection = self.clients[self.current_player]
                ptn = actions["pattern"]
                if actions["type"] == "chi":
                    chi_ptn = min(ptn) // 4
                    banned.append(tile_id // 4)
                    if tile_id // 4 == chi_ptn and chi_ptn % 9 != 7:
                        banned.append(chi_ptn + 3)
                    elif tile_id // 4 == chi_ptn + 2 and chi_ptn % 9 != 0:
                        banned.append(chi_ptn - 1)
                    self.game.chi(
                        player_id, ptn, kui_tile=tile_id, from_who=actions["from_who"]
                    )
                    logging.info(
                        blue(
                            f"「{self.clients[player_id].username}」chi「{' '.join(TENHOU_TILE_STRING_DICT[_] for _ in ptn)}」"
                        )
                    )
                elif actions["type"] == "pon":
                    banned.append(tile_id // 4)
                    self.game.pon(
                        player_id, ptn, kui_tile=tile_id, from_who=actions["from_who"]
                    )
                    logging.info(
                        blue(
                            f"「{self.clients[player_id].username}」pon「{' '.join(TENHOU_TILE_STRING_DICT[_] for _ in ptn)}」"
                        )
                    )
                elif actions["type"] == "kan":
                    ptn = [ptn[1] * 4 + i for i in range(4)]
                    self.game.kan(
                        player_id,
                        ptn,
                        mode=1,
                        kui_tile=tile_id,
                        from_who=actions["from_who"],
                    )
                    self.game.new_dora()
                    logging.info(
                        blue(
                            f"「{self.clients[player_id].username}」kan「{' '.join(TENHOU_TILE_STRING_DICT[_] for _ in ptn)}」"
                        )
                    )
                    tile_id, action = await self.handle_draw(who=player_id, where=-1)
                    tile_id, res = await self.handle_tsumo_action(tile_id, action)
                    if res:  # 自摸和了或者加杠被人抢和
                        if isinstance(res, list):
                            if len(res) == 3:
                                return {
                                    "event": "ryuukyoku",
                                    "why": "ron3",
                                    "action": res,
                                }
                            return res
                        event_type = res.get("type")
                        if event_type == "agari":
                            return [res]
                    after_tsumo = True
                continue
            else:
                self.current_player = (self.current_player + 1) % 4
                p = self.agents[self.current_player]
                connection = self.clients[self.current_player]

            if sum(self.game.kang_num) == 4 and 4 not in self.game.kang_num:
                return {"type": "ryuukyoku", "why": "kan4"}

            if self.game.left_num == 0:
                nagashimangan = [
                    i
                    for i in range(4)
                    if self.agents[i].nagashimangan
                    and all(
                        _ % 9 == 0 or _ % 9 == 8 or 27 <= _ <= 33
                        for _ in self.agents[i].discard_tiles
                    )
                ]
                machi_state = {
                    i: [list(self.agents[i].tiles), list(self.agents[i].machi)]
                    for i in range(4)
                    if self.agents[i].machi
                }
                return {
                    "type": "ryuukyoku",
                    "why": "yama_end",
                    "nagashimangan": nagashimangan,
                    "machi_state": machi_state,
                }

            tile_id, action = await self.handle_draw(who=self.current_player, where=0)
            tile_id, res = await self.handle_tsumo_action(tile_id, action)
            if res:  # 自摸和了或者加杠被人抢和
                if isinstance(res, list):
                    if len(res) == 3:
                        return {"event": "ryuukyoku", "why": "ron3", "action": res}
                    return res
                event_type = res.get("type")
                if event_type == "agari":
                    return [res]
                if event_type == "ryuukyoku":
                    return res
            p.ippatsu_status = 0  # 摸了牌没和，清除一发
            after_tsumo = True


class Server:
    def __init__(self, min_score, fast, train=False):
        self.ROOM_ID_LOCK = 0
        train = train and os.path.isfile("output/reward-model/checkpoints/best.pt")
        logging.debug("Training mode enabled")
        self.train = train
        self.game = GameEnvironment(
            has_aka=True, min_score=min_score, fast=fast, train=train
        )
    async def game_main_loop(self):
        self.game.game_start = True
        random.shuffle(self.game.clients)
        while self.game.game_start:
            self.game.start()
            wind = ["東", "南", "西", "北"][self.game.round // 4]
            wind_round = self.game.round % 4 + 1
            logging.info(
                green(
                    f"{wind}{wind_round} Round - {self.game.honba}本场------场供: {self.game.riichi_ba * 1000}"
                )
            )
            res = await self.game.game_loop()
            if res is None:
                logging.debug(yellow("Game Interruped..."))
                break
            if self.train:
                scores = [p.score for p in self.game.agents]
            game_over, score_delta = self.game.game_update(res)
            if self.train:
                for i in range(4):
                    self.game.reward_features[i].append(
                        torch.from_numpy(
                            self.game.game.get_game_feature(score_delta[i], scores[i])
                        )
                    )
                    for item in self.game.collected_data[i]:
                        if len(item) == 3:
                            continue
                        features = torch.stack(self.game.reward_features[i])[
                            None
                        ].float()
                        reward = self.game.reward(
                            features, len(self.game.reward_features[i]) - 1
                        )
                        item.append(reward)
            if not self.game.fast:
                await asyncio.sleep(2)
            if not self.game.fast:
                await asyncio.sleep(15)
            logging.info(green("Continue..."))
            if game_over:
                logging.info(green("Game Over！"))
                for i, score in self.game.game.get_rank():
                    logging.info(
                        green(
                            f"「{self.game.clients[i].username}」积分「{score * 100}」"
                        )
                    )
                self.game.game_start = False
                if not self.game.fast:
                    await asyncio.sleep(0.1)
                self.game.reset()
                break

    def game_thread(self):
        try:
            asyncio.run(self.game_main_loop())
        except Exception as e:
            tb = traceback.format_exc()
            logging.debug(red(f"An exception occurred: {e}"))
            logging.debug(red(f"Traceback info:\n{tb}"))

    async def run(self):
        while True:
            try:
                if len(self.game.clients) == 4 and not self.game.game_start:
                    thread = threading.Thread(target=self.game_thread, daemon=True)
                    thread.start()
                    thread.join()
                await asyncio.sleep(0.1)
            except Exception as e:
                tb = traceback.format_exc()
                logging.debug(red(f"An exception occurred: {e}"))
                logging.debug(red(f"Traceback info:\n{tb}"))
