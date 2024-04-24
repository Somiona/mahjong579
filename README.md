# mahjong579

- Train the discard model
```shell
$ python model/train_discard_model.py
```
- Train the riichi model
```shell
$ python model/train_riichi_model.py
```

- Train chi/pon/kan model
```shell
$ python model/train_furo_model.py --mode [chi / pon / kan]
```

## Self-Play
- Run the game with no wait time
```shell
$ python launcher.py -f
```