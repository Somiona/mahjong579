import unicodedata

from termcolor import colored


def get_visual_length(s):
    return sum(
        2 if unicodedata.east_asian_width(char) in ('F', 'W') else 1
        for char in s
    )


def pad_string(s, target_length):
    visual_length = get_visual_length(s)
    padding_length = target_length - visual_length
    if padding_length <= 0:
        return s[:target_length + (visual_length - target_length) // 2]
    left_length = padding_length // 2
    right_length = padding_length - left_length
    return ' ' * left_length + s + ' ' * right_length


def yellow(s):
    return colored(s, color='yellow', attrs=['bold', 'blink'])


def magenta(s):
    return colored(s, color='magenta', attrs=['bold', 'blink'])


def green(s):
    return colored(s, color='green', attrs=['bold', 'blink'])


def red(s):
    return colored(s, color='red', attrs=['bold', 'blink'])


def blue(s):
    return colored(s, color='blue', attrs=['bold', 'blink'])


def cyan(s):
    return colored(s, color='cyan', attrs=['bold', 'blink'])


def light_grey(s):
    return colored(s, color='light_grey', attrs=['bold', 'blink'])


chinese_numerals = {1: '一', 2: '二', 3: '三', 4: '四', 5: '五', 6: '六', 7: '七', 8: '八', 9: '九'}
TILE_STRING_DICT = {
    **{i: f"{chinese_numerals[i + 1]}萬" for i in range(9)},
    **{i: f"{chinese_numerals[i - 8]}饼" for i in range(9, 18)},
    **{i: f"{chinese_numerals[i - 17]}索" for i in range(18, 27)},
    27: "東", 28: "南", 29: "西", 30: "北", 31: "白", 32: "發", 33: "中"
}
TENHOU_TILE_STRING_DICT = {i: TILE_STRING_DICT[i // 4] for i in range(136)}
TENHOU_TILE_STRING_DICT[16] = '赤五萬'
TENHOU_TILE_STRING_DICT[52] = '赤五饼'
TENHOU_TILE_STRING_DICT[88] = '赤五索'

TILE_UNICODE = "🀇🀈🀉🀊🀋🀌🀍🀎🀏🀙🀚🀛🀜🀝🀞🀟🀠🀡🀐🀑🀒🀓🀔🀕🀖🀗🀘🀀🀁🀂🀃🀆🀅🀄︎"
TENHOU_TILE_UNICODE_DICT = {i: TILE_UNICODE[i // 4] for i in range(136)}
TENHOU_TILE_UNICODE_DICT[16] = '赤🀋'
TENHOU_TILE_UNICODE_DICT[52] = '赤🀝'
TENHOU_TILE_UNICODE_DICT[88] = '赤🀔'
