# nerf-reptile
### 0. 事前準備
#### 実行環境
pytorch  2.0.1+cu118
#### download data
以下のurlのsrn_cars.zipをダウンロードする。<br>
https://drive.google.com/file/d/19yDsEJjx9zNpOKz9o6AaK-E8ED6taJWU/view?usp=drive_link

[//]: # (https://drive.google.com/drive/folders/1IdOywOSLuK6WlkO5_h-ykr3ubeY9eDig)
### 1. train
```bash
python train.py
```
学習する際のオプションを変えたい場合は、args_parser.pyの数値を変更する。
