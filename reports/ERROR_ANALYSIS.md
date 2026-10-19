# 40 条 C 组 mixed 测试样例逐条文本复核（阶段 M5）

> 一句话结论：40条中12条规范化后零编辑错误、28条至少一个错误；主要残留是中文近音字（16）、英文词替换/拼写（12）、英文词边界（6）。`0%`输出崩溃率不等于语义正确：ID 400 的“并不需要→必须要”改变了否定含义。本复核是**AI文本对照，不是真人听音**。
>
> 相关文档：主结果与验收见 `FINAL_REPORT.md`；全量指标见 `RESULTS.md`；逐条机器可读原件见 `../artifacts/error_review_40.json`。

## 方法与边界
- 在训练前的协议中约定固定随机抽取40条；当前样本来自 `artifacts/error_review_40.json`，Random(42)，不按错误大小挑例。
- 本次由AI对参考/预测文本逐条复核，**不是真人听音标注**。没有听音的音近字、翻译及不清楚语音来源只标“疑似”，不伪装为声学证实。
- mixed 全测试集373条、5,566个参考计分单位；此40条用于归类和定位问题，不用于替代全量指标，也不据此推算错误发生率。
- 12条规范化后零编辑错误，28条至少一个编辑错误；4条参考含`[UNK]`，保留官方参考，不对未知区域擅自补标。
- 标签可复选，计数相加可能超过40。大小写和预设标点规范化不记错；语气词跨文字差异未额外清洗。

## 分类计数
| 标签 | 样例数 |
|---|---:|
| 英文词替换/拼写 | 12 |
| 英文词边界/分词 | 6 |
| 中文被英文替代 | 3 |
| 含UNK不确定 | 4 |
| 正确 | 12 |
| 中文词字错误 | 16 |
| 切换邻近冗字 | 1 |
| 英文被中文替代 | 2 |
| 语气词差异 | 6 |
| 英文增词 | 1 |
| 地名中文化（疑似翻译） | 1 |
| 否定含义错误 | 1 |

## 观察
1. 英文并非只会漏掉：privacy→private school、graduate school→credible school、McGrady→migratie 等替换/拼写/词边界错误仍存在。
2. opportunity→不经常 属于英文词消失并被中文替代，不能确认是语义翻译；Hong Kong→香港 属于地名中文化的疑似翻译案例，不能据此断言整个模型在执行翻译任务。
3. badminton→Bankman腾 是疑似音近字中文化；只凭文本不能证实发音成因。
4. 中文片段同样会错。尤其ID400的“并不需要→必须要”改变了否定含义；输出崩溃率为0%不代表语义可靠或可直接生产部署。
5. 只有“tough”后多出“的”等切换邻近冗字证据。没有听音及断句标注，**不能确认语言切换处断句错误的发生数**；未用标点变化冒充声学断句证据。
6. 官方mixed标签包含uh/um等极短语气词例；这不等同于每条都是实质性内容词的中英切换。主评测仍按官方标签，不事后改变子集。
7. 本节只分析C的残留错误，**不把单例或此40条当成C优于B的因果证据**；B/C主判断仍使用完整373条mixed的预设指标。

## 全部40条

表格读法：标签可复选，因此计数之和大于40；`[UNK]`是官方参考里的未知标记，不作补标；备注中的“疑似”表示仅凭文本无法确认声学原因。按测试ID可回到 `artifacts/error_review_40.json` 查看原始token IDs。
| 序号 / 测试ID | 参考 | C预测 | 标签 | 逐条备注 |
|---|---|---|---|---|
| 1 / 1134 | 我觉得python可能会用到一些 [UNK] neuralnetwork还有deep learning的东西 | you python可能会用到一些near narrow new row network还有 deep learning的东西 | 英文词替换/拼写、英文词边界/分词、中文被英文替代、含UNK不确定 | 我觉得→you；neuralnetwork→near narrow new row network。参考含[UNK]，不能把相邻全部英文增词都断定为幻觉。 |
| 2 / 291 | 太expensive | 太expensive | 正确 | 太expensive 与参考一致。 |
| 3 / 150 | 想过但是因为最近project有点多 | 想过但是因为这次project有点多 | 中文词字错误 | 最近→这次；project 保留。 |
| 4 / 494 | 所以你平时就现在大部分时间还是tennis就打网球 | 所以你平时就现在大部分时间还是tenness就打王球 | 英文词替换/拼写、中文词字错误 | tennis→tenness，网球→王球。 |
| 5 / 425 | 然后在打算申请graduate school的时候我考虑过挺多major的 | 然后在打算申请graduate school的时候我考虑过挺多major的 | 正确 | graduate school、major 和中文片段均保留。 |
| 6 / 404 | 对那我当初其实是我本科是e background | 那我当初其实是我本科是e background | 中文词字错误 | 句首对漏识别；e background 保留。 |
| 7 / 325 | 一开始的学习过程还是挺tough | 一开始的学习过程还是挺tough的 | 中文词字错误、切换邻近冗字 | tough 后额外生成的；不是可由当前文本确认的断句问题。 |
| 8 / 276 | 我我自己是之前的这边exchange就是我在这边当exchange students | 我我自己是之前在这边exchange就是我在这边当exchange student | 英文词替换/拼写、中文词字错误 | exchange students→exchange student，复数丢失；之前的这边→之前在这边。 |
| 9 / 1199 | 我当时学privacy的时候也是觉得 | 我当时学private school的时候也是觉得 | 英文词替换/拼写、英文词边界/分词 | privacy→private school，单词被替换为近形短语。 |
| 10 / 1035 | 但是如果你就是单纯的sell这个technology就是说告诉他就是我的我self driving这个方向是perfect | 但是如果你就是单纯的sell这个technology就是说告诉他就是我的我self driving这个function是perfect | 中文词字错误、中文被英文替代 | 方向→function；self driving、technology、perfect 保留。 |
| 11 / 256 | 就是会带我们组织一些activity | 就是会带我们组织一些activity | 正确 | activity 及中文上下文一致。 |
| 12 / 1076 | close [UNK] 嗯很多my friends in my generation | those keep很多my friends in my generation | 英文词替换/拼写、中文词字错误、含UNK不确定 | close→those，嗯漏掉；keep 对应[UNK]附近，无法只凭参考判其声学来源。 |
| 13 / 876 | 学生没有那种就是self awareness | 学生没有那种就是self awareness | 正确 | self awareness 保留。 |
| 14 / 188 | 深圳啊或者是上海这种比较大的城市会有更多opportunity | 深圳或者是上海这种表达城市会有更多的不经常 | 中文词字错误、英文被中文替代 | opportunity 消失，尾部变成不经常；比较大→表达。不是可确认的语义翻译，音近来源也未听音确认。 |
| 15 / 161 | uh呃 | so呃 | 语气词差异 | uh→so；极短语气词混合句，不能代表实质内容词切换。 |
| 16 / 264 | 所以你还蛮喜欢也蛮习惯Hong Kong life嘛就是 | 所以你还蛮喜欢也蛮习惯fongkong life什么就是 | 英文词替换/拼写、英文词边界/分词、中文词字错误 | Hong Kong→fongkong；life 保留；嘛→什么。 |
| 17 / 401 | 我要去这个university做交换 | 我要去这个university做交换 | 正确 | university 与中文上下文一致。 |
| 18 / 414 | 但是我的supervisor就跟我说也不是supervisor就是我的呃 | 那时候我的supervisor就跟我说也不是supervisor那时候我的呃 | 中文词字错误 | 两个但是/就是附近片段变成那时候；两个supervisor都保留。 |
| 19 / 999 | 就是hard choice | 就是hard choice | 正确 | 就是hard choice 一致。 |
| 20 / 1094 | so you will walk slowly and slowly and then you you you should呃 | so it will work slowly and slowly and then you you you should呃 | 英文词替换/拼写 | you→it，walk→work；末尾呃保留。 |
| 21 / 151 | 六月一号两门final吗 | 六月一号我的两门final嘛 | 中文词字错误、语气词差异 | 额外生成我的，吗→嘛；final 保留。 |
| 22 / 1047 | 第二个呢它的测试也很很复杂就是说你在测试的时候你只能simulate这个scenario | 第二个呢它的测试也很很复杂就是说你在测试的时候你只能simulate这个scenario | 正确 | simulate、scenario 及中文片段一致。 |
| 23 / 388 | not what I expected in the beginning but just对最后就回到这里 | it's not what I expected in the beginning but just对最后就回到这里 | 英文增词 | 句首多出it's；原参考为not what I expected...。 |
| 24 / 1284 | mobile phone number还有你的嗯 | mobile phone number还有你的嗯 | 正确 | mobile phone number 及中文片段一致。 |
| 25 / 1145 | 哦stock吗对有 | oh stock吗对有 | 语气词差异 | 哦→oh，发音/语义可能近似，但冻结计分不做跨文字语气词归一化。 |
| 26 / 1259 | credit card或者是octopus card这种东西 | credit card或者是octobus card这种东西 | 英文词替换/拼写 | octopus→octobus；credit card 保留。 |
| 27 / 863 | 校服上面没有学校的logo吗 | 校服上面没有学校的logo吗 | 正确 | logo 及中文片段一致。 |
| 28 / 402 | 所以但是跟那这跟major没关系的你说的major可能是哦我接下来要apply到graduate school的时候我要往computer science和往electric engineering | 所以但是跟这个major没关系的你说的major可能是哦我接下来要apply到credible school的时候我要往tablet of science或者往electric engineering | 英文词替换/拼写、英文词边界/分词、中文词字错误 | graduate school→credible school，computer science→tablet of science，另有中文替换。 |
| 29 / 937 | 嗯这边的schedule每天都排的很满然后很紧 | 嗯这边的schedule每天都拍得很慢然后很紧 | 中文词字错误 | 排的很满→拍得很慢，存在近音字误识别；schedule 保留。 |
| 30 / 1071 | 就是 [UNK] surfing the internet | 就是给开server in the internet | 英文词替换/拼写、英文词边界/分词、含UNK不确定 | surfing the internet→server in the internet；给开 位于[UNK]附近，无法直接判其来源。 |
| 31 / 554 | 嗯所以你平时会看Olympic games吗过去的几届 | 嗯所以你平时会看olympic games吗过去的季节 | 中文词字错误 | 几届→季节；Olympic games 的大小写差异不计错。 |
| 32 / 55 | um运动啊男生一般不都喜欢运动 | 嗯运动啊男生一般不都喜欢运动 | 语气词差异 | um→嗯；按官方标签保留为mixed，但实质是语气词的跨文字标注差异。 |
| 33 / 346 | 去念研究所那为什么选择来Hong Kong | 去念你就拿卫生学的来香港 | 中文词字错误、地名中文化（疑似翻译） | Hong Kong→香港，参考中的英文地名被中文化；前段中文同时严重误识别，不能据此断言整句被翻译。未做听音确认。 |
| 34 / 1254 | 香港他们很注重privacy的问题 | 香港他们很注重privacy的问题 | 正确 | privacy 及中文片段一致。 |
| 35 / 681 | 比较容易学的是swim | 比较容易学的是swim | 正确 | swim 及中文片段一致。 |
| 36 / 342 | 如果现在有一个呃问题问一下就是why why did you come to Hong Kong | 如果现在有一个呃问题问一下就是why why did you come to hong kong | 正确 | Hong Kong→hong kong 仅大小写变化，规范化后完全一致。 |
| 37 / 400 | 就你要申请去交换你并不需要去说哦我要去哪一个department交换是直接选school | 就你要升起去交换你必须要去说哦我要去哪一个department交换是situation school | 中文词字错误、中文被英文替代、否定含义错误 | 申请→升起，并不需要→必须要，否定含义改变；直接选→situation。department、school 保留。 |
| 38 / 677 | badminton是一个比较好学的就是说 | Bankman腾是一个比较好学的就是说 | 英文词替换/拼写、英文被中文替代 | badminton→Bankman腾，部分英文被中文音近字替代的疑似案例；没有听音，不能确认声学原因。 |
| 39 / 719 | 对然后当时that's the reason I know the [UNK] um the famous player McGrady | 对然后当时that's the reason i know themigrate呃the famous player migratie | 英文词替换/拼写、英文词边界/分词、语气词差异、含UNK不确定 | McGrady→migratie，专名拼写/边界错误；um→呃。[UNK]附近的themigrate无法逐词可靠判定。 |
| 40 / 267 | 啊lifestyle啊之类的 | 呃lifestyle啊之类的 | 语气词差异 | 首个啊→呃；lifestyle 保留。 |

机器可读复核结果见 `evidence/checks/error_analysis_40.json`；原始 token IDs 与崩溃标记保留在C完整预测及抽样原件中。
