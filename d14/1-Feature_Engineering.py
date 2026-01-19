#%%
 from pandas.core.arrays.sparse.scipy_sparse import sparse_series_to_coo
from sklearn.feature_extraction import DictVectorizer   #feature_extraction特征抽取包
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer   #text文本特征抽取包
from sklearn.preprocessing import MinMaxScaler, StandardScaler #minmaxscaler是归一化，standardScaler是标准化
from sklearn.feature_selection import VarianceThreshold #特征选择
from sklearn.decomposition import PCA #主成分分析
import jieba
import numpy as np
from sklearn.impute import SimpleImputer
#%% md
# # 1 特征中含有字符串的时候（当成类别），如何做特征抽取
#%%

def dictvec():
    """
    字典数据抽取
    :return: None
    """
    # 实例化
    # sparse改为True,输出的是每个不为零位置的坐标，稀疏矩阵可以节省存储空间
    #矩阵中存在大量的0，sparse存储只记录非零位置，节省空间的作用
    #Vectorizer中文含义是矢量器的含义：DictVectorizer() 是 scikit-learn 中用于将字典格式的数据转换为数值特征向量的工具：主要作用： 将包含特征名和特征值的字典列表转换为机器学习算法可以使用的数值矩阵
    dict1 = DictVectorizer(sparse=True)  # 把sparse改为True看看，sparse=True为默认值，因为现实中的数据大多是稀疏的
                                           #创建了一个对象，用以将字典数据转化为特征向量，返回的是稀疏矩阵形式
    dict2 = DictVectorizer(sparse=False)
    data_false = dict2.fit_transform([{'city': '北京', 'temperature': 100},
                               {'city': '上海', 'temperature': 60},
                               {'city': '深圳', 'temperature': 30}])
    print(data_false)
    #每个样本都是一个字典，有三个样本
    # 调用fit_transform
    data = dict1.fit_transform([{'city': '北京', 'temperature': 100},
                               {'city': '上海', 'temperature': 60},
                               {'city': '深圳', 'temperature': 30}])    #将返回的稀疏矩阵进行拟合并转换，fit_transform是一个组合方法，
                                                                        # fit（）学习并拟合：从数据中学习参数如特征名称，均值，标准差等，  transform（）进行转换，使用已经学习的数据的参数转换数据
    print(data)
    print('-' * 50)
    # 字典中的一些类别数据，分别进行转换成特征
    print(dict1.get_feature_names_out())
    print('-' * 50)
    print(dict1.inverse_transform(data))  #去看每个特征代表的含义，逆转回去,方便人类看一下

    return None

#one——hot编码：将每个类别值转换为一个二进制向量，只有一个位置是 1（hot），其余都是 0
#稀疏编码的coords表示非零位置的坐标，values表示非零位置的值,稀疏编码不能直接喂给机器学习模型，需要转换成密集矩阵才行
dictvec()
#%% md
# # 2 一段英文文本如何变为数值类型
#%%

def couvec():
    # 实例化CountVectorizer
    # max_df, min_df整数：指每个词的所有文档词频数不小于最小值，出现该词的文档数目小于等于max_df
    # max_df, min_df小数(0-1之间的）表示：某个词的出现的次数／所有文档数量
    # min_df=2表示出现次数小于2的词不统计
    # 默认会去除单个字母的单词，默认认为这个词对整个样本没有影响,认为其没有语义
    vector = CountVectorizer(min_df=2)  #统计每个单词在每个文档中出现的次数，将文本转换为数值特征，返回的是稀疏矩阵

    # 调用fit_transform输入并转换数据

    res = vector.fit_transform(
        ["life is  short,i like python life",
         "life is too long,i dislike python",
         "life is short"])

    # 打印结果,把每个词都分离了
    print(vector.get_feature_names_out()) #获取特征名称
    print('-'*50)
    print(res)
    print('-'*50)
    print(type(res))
    # 对照feature_names，标记每个词出现的次数
    print('-'*50)
    print(res.toarray()) #稀疏矩阵转换为数组
    print('-'*50)
    #拿每个样本里的特征进行显示
    print(vector.inverse_transform(res))


couvec()
#%% md
# # 一段汉字文本如何数值化，对于汉字不能用空格来分割
#%%

def countvec():
    """
    对文本进行特征值化,单个汉字单个字母不统计，因为单个汉字字母没有意义
    :return: None
    """
    cv = CountVectorizer()

    data = cv.fit_transform(["人生苦短，我 喜欢 python python", "人生漫长，不用 python"])

    print(cv.get_feature_names_out())
    print('-'*50)
    print(data) #稀疏存储，只记录非零位置
    print('-'*50)
    print(data.toarray())

    return None


countvec()
#%% md
# ## 1.3 掌握如何对中文进行分词
#%%

def cutword():
    """
    通过jieba对中文进行分词，jieba是一个开源的分词库
    :return:
    """
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")

    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")

    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    # 转换成列表
    print(type(con1))  #generator类型
    print('-' * 50)
    # 把生成器转换成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)
    print(content1)
    print(content2)
    print(content3)
    # 把列表转换成字符串,每个词之间用空格隔开
    print('-' * 50)
    c1 = ' '.join(content1)    #转换成空格分隔的字符串
    c2 = ' '.join(content2)    #jieba是使用的传统的分词算法，支持自定义分词，eg：it领域的词典；还可以设置停用词
    c3 = ' '.join(content3)    #用空格连起来，先用jieba把词分出来，再用空格连接起来，jieba类似一个前置工作
                               #英文文本是天然空格分隔的，中文需要手动添加空格
    return c1, c2, c3          #sklearn的文本处理工具（如CountVectorizer）默认按空格分词


def hanzivec():
    """
    中文特征值化
    :return: None
    """
    c1, c2, c3 = cutword() #jieba分词好的中文文本，这行代码调用了 cutword() 函数，获取三个分词后的中文文本字符串
    print('-'*50)
    print(c1)
    print(c2)
    print(c3)
    print('-'*50)

    cv = CountVectorizer()

    data = cv.fit_transform([c1, c2, c3])  #学习并转换数据

    print(cv.get_feature_names_out())

    print(data.toarray())

    return None

# cutword()
hanzivec()
#%% md
# # 1.4 tf-idf
#%%
# 规范{'l1'，'l2'}，默认='l2'
# 每个输出行都有单位范数，或者：
#
# 'l2'：向量元素的平方和为 1。当应用 l2 范数时，两个向量之间的余弦相似度是它们的点积。
#
# 'l1'：向量元素的绝对值之和为 1。参见preprocessing.normalize。

# smooth_idf布尔值，默认 = True
# 通过在文档频率上加一来平滑 idf 权重，就好像看到一个额外的文档包含集合中的每个术语恰好一次。防止零分裂。
# 比如训练集中有某个词，测试集中没有，就是生僻词，就会造成n(x)分母为零，log(n/n(x)),从而出现零分裂

def tfidfvec():
    """
    中文特征值化,计算tfidf值
    :return: None
    """
    c1, c2, c3 = cutword()

    print(c1, c2, c3)
    # print(type([c1, c2, c3]))
    tf = TfidfVectorizer(smooth_idf=True)  #smooth_idf=True表示平滑处理,具体公式是log(1+df/dt)+1

    data = tf.fit_transform([c1, c2, c3])

    print(tf.get_feature_names_out())
    print('-'*50)
    print(type(data))
    print('-'*50)
    print(data.toarray())

    return None


tfidfvec()
#%% md
# # 2 特征处理，不同的特征拉到到同一个量纲
#%%
def mm():
    """
    归一化处理，按列进行归一化处理
    :return: NOne
    """
    # 归一化缺点 容易受极值的影响
    #feature_range代表特征值范围，一般设置为(0,1),或者(-1,1),默认是(0,1)
    mm = MinMaxScaler(feature_range=(0, 1))

    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])

    print(data)
    print('-'*50)
    out=mm.transform([[1, 2, 3, 4],[6, 5, 8, 7]])
    print(out)
    return None
    #transform和fit_transform不同是，transform用于测试集，而且不会重新找最小值和最大值


mm()
#%%
(1-60)/30
#%%
def stand():
    """
    标准化缩放，不是标准正太分布，只均值为0，方差为1的分布
    :return:
    """
    std = StandardScaler()

    data = std.fit_transform([[1., -1., 3.], [2., 4., 2.], [4., 6., -1.]])

    print(data)
    print('-' * 50)
    print(std.mean_) #均值
    print('-' * 50)
    print(std.var_) #方差
    print(std.n_samples_seen_)  # 样本数
    return data


data=stand()
#%%
type(data)
print(data)
#%%
std1 = StandardScaler()
#为了证明上面输出的结果的均值是为0的，方差为1
data1 = std1.fit_transform([[-1.06904497, -1.35873244,  0.98058068],
 [-0.26726124,  0.33968311,  0.39223227],
 [ 1.33630621,  1.01904933, -1.37281295]])   #利用这个fit_transform()这个接口会自动计算均值和标准差
#print(data1)  #这个并不是我们想看的，没意义
# 均值
print(std1.mean_)
# 方差
print(std1.var_)
#%%
(np.square(1-2.333)+np.square(2-2.333)+np.square(4-2.333))/3#np.square表示平方，这行代码在计算方差
#%%
(1-2.333)/np.sqrt(1.55555)    #这行代码在计算标准差，np.sqrt()计算平方根
#%% md
# ## transform和fit_transform不同是，transform用于测试集，而且不会重新找最小值和最大值,不会重新计算均值方差，只代入数据
#%% md
# # 3 缺失值处理
# 
#%%
#下面是填补，针对删除，可以用pd和np
def im():
    """
    缺失值处理
    :return:NOne
    """
    # NaN, nan,缺失值必须是这种形式，如果是？号(或者其他符号)，就要replace换成这种
    im = SimpleImputer(missing_values=np.nan, strategy='mean')

    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6], [3, 2]])

    print(data)

    return None


im()
#%%
11/3
#%%
np.nan+10
#%% md
# # 4 降维
# # 降维就是特征数变少
# # 降维可以提高模型训练速度（模型的参数减少了）
#%%
def var():
    """
    特征选择-删除低方差的特征
    :return: None
    """
    #默认只删除方差为0,threshold是方差阈值，删除比这个值小的那些特征，
    var = VarianceThreshold(threshold=0.1)

    data = var.fit_transform([[0, 2, 0, 3],
                              [0, 1, 4, 3],
                              [0, 1, 1, 3]])

    print(data)
    # 获得剩余的特征的列编号，True表示只显示剩余的特征的列编号，
    print('The surport is %s' % var.get_support(True))
    return None


var()

#%%
def pca():
    """
    主成分分析进行特征降维
    :return: None
    """
    # n_ components:小数 0~1 90% 业界选择 90~95%

    # 当n_components的值为0到1之间的浮点数时，表示我们希望保留的主成分解释的方差比例。方差比例是指 得到输出的每一列的方差值和除以原有数据方差之和。
    # 具体而言，n_components=0.9表示我们希望选择足够的主成分，以使它们解释数据方差的90%。

    # n_components如果是整数   减少到的特征数量
    # 原始数据方差
    original_value = np.array([[2, 8, 4, 5], #特征维数是4
                               [6, 3, 0, 8],
                               [5, 4, 9, 1]])
    print(np.var(original_value, axis=0).sum()) #最初数据每一列的方差，求和
    print('-'* 50)
    pca = PCA(n_components=0.9)  #搞一个对象，参数表示保留大于等于90%的方差，0.9是业界的经验

    data = pca.fit_transform(original_value)

    print(data)
    print(type(data))
    #计算data的方差
    print(np.var(data, axis=0).sum())
    print('-'*50)
    print(pca.explained_variance_ratio_)
    # 计算data的方差占总方差的比例
    print(pca.explained_variance_ratio_.sum())

    return None


pca()
#%%
22/29.333333333333336
#%%
from matplotlib import pyplot as plt
x = np.random.rand(10000) #每个的概率，random.rand()是生成0到1之间的随机数
t = np.arange(len(x))
plt.plot(t,x,'g.',label="Uniform Distribution")
plt.legend(loc="upper left")
plt.grid()
plt.show()