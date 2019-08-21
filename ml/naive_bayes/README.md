# 机器学习算法：朴素贝耶斯

朴素贝叶斯是比较基础的一种机器学习算法，与依托梯度训练的常规机器学习算法不同，朴素贝叶斯主要是基于统计概率学来进行预测的。

## 贝叶斯定理
贝叶斯定理是朴素贝叶斯算法使用的核心，也是其名字的来源。贝叶斯定理可以写作如下：
$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$
贝叶斯定理的推导非常简单，我们来看这个恒等式：$P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$
用直觉来理解就是，要计算A和B两件事同时发生的概率，我们可以用两种分步的方式来计算。为了方便理解，这里我们假定一个场景，我们网上认识了一个妹子，声音很甜，但是我们想知道这个妹子是不是长得也很漂亮，则我们可以设B是妹子声音很甜，A是妹子颜值很高。想要计算既妹子声音很甜又妹子颜值很高的概率，我们可以先算妹子声音很甜的概率，再算妹子声音很甜的情况下妹子颜值很高的概率，然后两者相乘，也可以先算妹子颜值很高的概率，再算妹子颜值很高的情况下妹子声音很甜的概率，然后两者相乘。将这个式子的$P(B)$项移动到等式右边，我们就得到了贝叶斯定理。
这其中$P(A|B)$我们称为后验概率（也就是我们想求的概率，在妹子声音很甜的情况下，颜值也高的概率），$P(A)$称为先验概率（妹子声音很甜），$P(B|A)$称为条件概率（在妹子很美的前提下，声音也甜的概率）。（这些是概念，不理解也没问题）

## 为何叫朴素
在上面的例子中，我们的情况相对简单，条件只有一个，就是妹子声音很甜。在朴素贝叶斯的使用场景中，条件一般要复杂得多，比如我们可以设定我们知道了妹子是否是吃货、婚姻情况、是否喜欢动漫、是否爱打游戏、是否喜欢汉服、声音是否好听这几个条件下，使用朴素贝叶斯来判断妹子是否好看。也就是说，B通常是多个事件的联立，我们想求的式子可以写成：
$P(好看|吃甜、未婚、喜欢动漫、不爱游戏、喜欢汉服、声音好听) = \frac{P(吃甜、未婚、喜欢动漫、不爱游戏、喜欢汉服、声音好听|好看) \cdot P(好看)}{P(吃甜、未婚、喜欢动漫、不爱游戏、喜欢汉服、声音好听)}$
在这里，我们从收集的妹子数据样本中可以很容易地得到妹子好看的概率$P(好看)$，但是$P(吃甜、未婚、喜欢动漫、不爱游戏、喜欢汉服、声音好听|好看)$这一项不知道怎么求，我们在朴素贝叶斯中，有一个假设前提，就是这些联立事件之间是独立发生的，也就是说妹子吃不吃甜和妹子喜不喜欢汉服、是否已婚等都是互不干扰的，这样的话，$P(吃甜、未婚、喜欢动漫、不爱游戏、喜欢汉服、声音好听|好看)$就可以写成$P(吃甜|好看)P(未婚|好看)P(喜欢动漫|好看)P(不爱游戏|好看)P(喜欢汉服|好看)P(声音好听|好看)$，因为独立事件之间的概率可以直接相乘。这样一来，这每一项项我们都可以通过样本计算出来，分子就是完全可以求出来的了。
分母上的这个概率非常难求，但是我们不用纠结它，我们再求一次妹子在当前条件下不好看的概率：$P(不好看|吃甜、未婚、喜欢动漫、不爱游戏、喜欢汉服、声音好听) = \frac{P(吃甜、未婚、喜欢动漫、不爱游戏、喜欢汉服、声音好听|不好看) \cdot P(不好看)}{P(吃甜、未婚、喜欢动漫、不爱游戏、喜欢汉服、声音好听)}$，就会发现其分母还是这个分母，因此我们比较妹子好看和不好看的概率时，直接比较两者的分子就可以了，然后哪种情况的概率高，我们就预测哪种情况。
在这里，各联立事件之间是独立的这个假设虽然牺牲了一些预测精度，却简化了朴素贝叶斯的求解，让整个算法的计算变得可行，而我们称其朴素，也正是因为这个独立的假设。

## 算法实践
朴素贝叶斯的训练过程实质上是计算先验概率和条件概率的过程，假设我们现在有1000个妹子的喜好样本以及她们是否好看的标签，那么先验概率$P(不好看)$与$P(好看)$很容易计算，用好看与不好看的个数除以1000即可。
对于条件概率的计算，我们假设想要计算$P(声音好听|不好看)$，只要统计一下有多少个妹子不好看，放在分母，再统计一下这些妹子中有多少声音好听，放在分子，得到的结果就是对应的条件概率。
将这些先验概率和条件概率都计算完毕以后，我们要预测一个新样本的，只要将其代入朴素贝叶斯公式算出各种标签下的分子，然后预测分子最大对应的标签即可。

## 缺失值的处理
在我们预测样本的时候，有一个缺陷，因为我们的计算公式是这样的：
$P(好看|吃甜、未婚、喜欢动漫、不爱游戏、喜欢汉服、声音好听) = \frac{P(吃甜|好看)P(未婚|好看)P(喜欢动漫|好看)P(爱游戏|好看)P(喜欢汉服|好看)P(声音好听|好看)}{P(吃甜、未婚、喜欢动漫、不爱游戏、喜欢汉服、声音好听)}$
在分子上是各个概率的乘积，假如在样本中好看的妹子里面，一个爱打游戏的妹子都没有，那么P(爱游戏|好看)就为0，则整个分子一定是0，其他的概率项就不用看了，这是明显不太合理的。在这里我们用拉普兰斯正则化来处理这样条件概率为0的情况，在求条件概率的时候，给分子的值加1，给分母的值加上N，N为样本中特征值的总量，在我们这个例子里N为6，那么在原来P为0的时候，正则化后其概率为$\frac{1}{6}$，避免了一个条件概率为0，整个分子都为0的不合理情况。

## 代码实现
```python
class NaiveBayesClassifier:

    def __init__(self):
        self.prior = None
        self.conditional = dict()
        self.labels = None
        self.min_prob = 0

    def fit(self, X, Y):
        total_samples = len(Y)

        X = X.tolist()
        Y = Y.tolist()
        # 计算先验概率
        label_count = Counter(Y)
        self.labels = list(label_count.keys())
        self.prior = {key: value/total_samples for key, value in label_count.items()}

        # 计算条件概率
        features_count = Counter(feature for sample in X for feature in sample)
        self.min_prob = 1 / len(features_count)

        # initialization
        for label in label_count.keys():
            self.conditional[label] = dict()
            for feature in features_count.keys():
                self.conditional[label][feature] = 0

        # count all instances
        for i, sample in enumerate(X):
            for feature in sample:
                self.conditional[Y[i]][feature] += 1

        # calculate probability
        for label in label_count.keys():
            for feature in features_count.keys():
                # laplace normalization
                self.conditional[label][feature] = (self.conditional[label][feature]+1) / (label_count[label]+len(features_count))

    def predict(self, X):
        X = X.tolist()
        predictions = []
        for sample in X:
            sample_pred = []
            for label in self.labels:
                product_of_cond = 1
                for feature in sample:
                    product_of_cond *= self.conditional[label].get(feature, self.min_prob)
                sample_pred.append(product_of_cond * self.prior[label])
            predicted_label = self.labels[np.argmax(np.array(sample_pred))]
            predictions.append(predicted_label)
        return predictions

```