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

## 環境構築手順(CentOSやAmazon Linux)の場合
To be revised.<br>
<br>
https://gym.openai.com/docs/<br>

