# ATEC_Payment_Risk_Recognition_Competition
ATEC蚂蚁开发者大赛-风险大脑-支付风险识别

### 数据集介绍
正样本数量: 12122
负样本数量: 977884
未标记样本数量: 4725

## 结果
### 1.submission_180514_v4_0.1553.csv 

1. 使用的是过采样了的样本;
2. 空值填充为众数;
3. 特征向量未进行归一化处理;
4. 训练LightGBM进行分类;
5. 将特征分量中unique值数量小于等于10的都设置为category类型;
6. 使用5-fold进行训练(即: train: 0.8*0.85, valid: 0.8*0.15, test: 0.2);
7. 最后用在测试集时, 使用训练得到的5个模型进行预测加权;

LightGBM的参数如下所示:

	lgb_params = {'boosting_type': 'gbdt',
                      'num_leaves': 31,
                      'max_depth': 10,
                      'learning_rate': 0.10,
                      'n_estimators': 100,
                      'reg_alpha': 0.1,
                      'seed': 42,
                      'nthread': -1}

训练集上的结果:
	
		P         R        F1       AUC      mayi
	0  0.969269  0.986218   0.97767  0.995363  0.614086
	1  0.969314  0.986065  0.977618  0.995428  0.618365
	2  0.969256  0.986427  0.977766  0.995377  0.614159
	3  0.969213  0.985972  0.977521  0.995342  0.616181
	4  0.969326  0.986336  0.977757  0.995457  0.618051

测试集上的结果:

		P         R        F1       AUC      mayi
	0  0.969192  0.986074   0.97756  0.995267  0.601931
	1  0.969475  0.985699  0.977519  0.995362  0.609117
	2  0.968328  0.985653  0.976914  0.995136  0.592778
	3  0.969159  0.985617  0.977319  0.995086  0.589465
	4  0.968965  0.986202  0.977508  0.995327  0.614417

AUC快达到1了,严重过拟合了吧. 不过在测试集上的表现竟然和在训练集上差不多, 这说明了数据的分布比较集中啊!

将LGB模型的n_estimators参数设置为20, 其他保持不变, 得到的结果是:
训练集:

          P         R        F1       AUC      mayi
	0    0.9553  0.964216  0.959737  0.989568  0.594197
	1  0.955233   0.96511  0.960146  0.989536  0.459047
	2  0.955267  0.964727  0.959974  0.989581  0.456871
	3  0.954816   0.96381  0.959292  0.989417  0.459641
	4  0.954342  0.965354  0.959817  0.989311   0.60242

测试集:
          P         R        F1       AUC      mayi
	0  0.955472  0.964712   0.96007  0.989541  0.472286
	1  0.955552  0.965114  0.960309  0.989655  0.600039
	2  0.954136  0.964355  0.959218  0.989288  0.446783
	3  0.955325   0.96308  0.959187  0.989469  0.457052
	4  0.954786  0.965735  0.960229  0.989403  0.603666

结论:
1. 数据集这些数据分布太密了, 感觉过采样这种方式可能不得行!

### 2.submission_180510_v1_0.2904.csv

1. 对缺失值使用最小值来填充;
2. 训练K个RF分类器, 对于每个分类器, 训练样本为: 所有正样本+3倍的随机抽取的负样本;
3. 测试时, 使用这K个分类器得到的结果进行加权;

分类器的参数:

	rf = RandomForestClassifier(n_estimators=20,
                                    max_depth=10,
                                    n_jobs=-1)

trainset上:

          P         R        F1       AUC      mayi
	0  0.923779  0.907792  0.915716  0.989913  0.663437
	1   0.91082  0.911821  0.911321  0.989445  0.663144
	2  0.908522  0.916674   0.91258  0.989602  0.659308
	3   0.92641  0.902573  0.914336  0.990276  0.667732
	4  0.930508  0.904862  0.917506  0.990232  0.638302

验证集上:
	
          P         R        F1       AUC      mayi
	0  0.913007  0.900083  0.906499  0.988728  0.621565
	1  0.875207  0.881765  0.878474  0.979728  0.391341
	2  0.875702  0.909242  0.892157  0.985488  0.618984
	3  0.912786  0.897585  0.905122  0.986607  0.456619
	4  0.905612  0.886761  0.896088  0.983932  0.407577


训练集日期范围是: 2017.09.05 - 2017.11.05
测试集日期范围是: 2018.01.05 - 2018.01.05

训练集中, 每天对应的正样本数量是150左右, 负样本数量是15000左右;
如果按照日期来抽样样本会怎么样呢?


## Submission V5 (0.3413)
**样本方面**: (1) 每个模型使用了所有正样本和随机抽取的三倍负样本(按照每个date进行分组了); (2) 对每个特征分量的空值使用了其最小值进行填充; 

**模型方面**: 训练了5个模型进行加权:

        models = [
            RandomForestClassifier(n_estimators=20, max_depth=10, n_jobs=-1),
            ExtraTreesClassifier(n_estimators=20, max_depth=10, n_jobs=-1),
            #AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), algorithm="SAMME", n_estimators=10),
            LGBMClassifier(n_estimators=20, max_depth=10),
            GradientBoostingClassifier(n_estimators=15, max_depth=10),
            XGBClassifier(n_estimators=15, max_depth=8, n_jobs=-1)
            ]

**模型使用**: 先投票, 得到样本被分成了正/负. 然后排除掉少类的评委, 对多类评委的评分进行求平均;


## Submission V6 (2018.05.17 18:03)
**改进**:
(1) **样本方面**: 使用date分组后,每组中正样本数量为100+,负样本数量为10000+; 可不可以对负样本使用KMeans聚类, 选出那些具有代表性的负样本进行训练? (可能要耗费更多时间!)
(2) **其他**: 其他先不变了!


(1) 模型训练了1:18:36, 943s/个
训练集上准确率:

	          P         R        F1       AUC      mayi
	0  0.916301   0.91363  0.914964   0.98907  0.629528
	1   0.90729  0.780157  0.838934  0.976634   0.47459
	2  0.898059  0.889304   0.89366  0.982736  0.495599
	3  0.941703  0.938958  0.940328  0.991101  0.737919
	4  0.913719  0.905977  0.909831  0.987231   0.64139

测试集上准确率:

          P         R        F1       AUC      mayi
	0  0.900616  0.893543  0.897065  0.984621  0.471204
	1  0.896378  0.777487   0.83271  0.973358  0.338569
	2  0.887055  0.890925  0.888986  0.979436  0.393979
	3  0.904393   0.91623  0.910273  0.986026  0.555497
	4   0.89102  0.891798  0.891409  0.985218  0.474695

(2) 模型已经保存下来了(**model_v6.npz**)

(3) 结果保存为: submission_180517_v6.csv

(4) 最终评测结果: 0.3175, 反而比v5降低了!











