# kaggle_titanic
predict titanic survived
电影《泰坦尼克号》是由真实事件改编的一部电影，讲述的是1912年4月10日，泰坦尼克号的处女航从英国南安普敦出发，计划驶往美国纽约。但在4月14日深夜，这艘堪称“世界上最大的不沉之船”不幸撞上北大西洋上漂浮的冰山而沉没，船上2208名旅客中，仅有705人生还，生还率不到32%，这也成为人类史上最大的海难之一。
导致这么多人遇难的原因之一是乘客和船员没有足够的救生艇。虽然幸存下来的人有运气成分存在，但有一些人比其他人更有可能生存，比如妇女、儿童和上层阶级。沉船前，除了杰克和露丝的爱情外，泰坦尼克号还经历了什么？
所以我们研究的问题是：什么样的人在泰坦尼克号中更容易存活？

"Embarked“ 填充S
将S,C,Q分别转化为0，1，2

特征创建 计算家庭人口数
FamilySize = SibSp（兄弟姐妹数） + Parch（父母子女数）

提取name中的称呼"Mr", "Mrs", "Miss", "Master", "Dr", "Rev"等
同Embarked变为数字

对所有数据属性标准化

使用集成学习
第一层
SVC，KNeighborsClassifier，DecisionTreeClassifier，BaggingClassifier，RandomForestClassifier
第二层用XGBClassifier

使用GridSearchCV调整参数
