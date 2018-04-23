# reinforcement_learning
強化学習の勉強用<br>
<br>
※ このブランチ[continue_to_upload_until_may]は5月までの期間限定。マスターを更新しないように設けたブランチ。

## 環境
python 3<br>
OpenAI Gym<br>
<br>
2だと動かなかったりする。<br>
(自分の環境ではImportError: cannot import name 'spaces'が出て解決できず)<br>
<br>
動く場合もあるが"おまじない"が必要な場合。<br>
Spaces.Box呼出箇所にて、Spaces.Boxの引数dtypeの指定を削除する。<br>
Spaces.Boxのコンストラクタ(init)の引数が2と3で違うらしい。<br>
(2はinitの引数にdtypeが無い)<br>

## 環境構築手順
公式の手順は下記。<br>
https://gym.openai.com/docs/

### Ubuntu16,17
以下のファイルを削除する(削除しないとaptコマンドでエラーが出る場合がある)
```shell
$ sudo rm /var/lib/apt/lists/lock
$ sudo rm /var/lib/dpkg/lock
```
gitとpip3をインストールする。
```shell
$ sudo apt install python3-pip git
```
OpenAI Gymをインストールする。
```shell
$ pip3 install gym
```
Tensoflowをインストールする(最新版の1.7だとエラーが出たので1.5指定)。
```shell
$ pip3 install tensorflow==1.5.0
```
Kerasをインストールする。
```shell
$ pip3 install keras
```
作業用のフォルダを作成する(名前はなんでも良いです)
```shell
$ mkdir workspace
$ cd workspace
```
GitHubからソースをダウンロードする。
```shell
$ git clone https://github.com/indspug/reinforcement_learning
```
yusuke_1goの実行方法。
```shell
$ cd reinforcement_learning/02.yusuke_1go
$ python3 main.py
```
