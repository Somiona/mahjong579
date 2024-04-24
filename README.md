# mahjong579

- 训练弃牌模型
```shell
$ python sl_train/train_discard_model.py --num_layers 50 --epochs 10
```
- 训练立直模型
```shell
$ python sl_train/train_riichi_model.py --num_layers 20 --epochs 10
```

- 训练副露模型
```shell
$ python sl_train/train_furo_model.py --mode chi --num_layers 50 --epochs 10 --pos_weight 10
```

## Self-Play
目前只做了环境，并没有加入任何强化学习的逻辑
```shell
$ python online_game/server.py -A 4 -f  # -f参数开启快速模式，跳过所有AI思考时间和等待时间
$ python online_game/server.py -A 4 -d -ob  # -ob参数开启观战模式(不建议在-f模式下进行观战...)

$ python online_game/client.py -ob "一姬1(简单)"  # 观战某个玩家（现在可以用下面提供的网页版客户端来观战啦～）
```