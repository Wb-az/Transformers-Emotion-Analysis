
<h1><div style="text-align: center;"> Emotion analysis of tweets with customised transformers classifiers </div></h1>
<h3><div style="text-align: center"> BERT, DistilBERT & RoBERTa </div></h3>


<h2><div>1. Introduction</div></h2>

Natural language processing (NLP), is a branch of 
artificial intelligence involved with human and computer interactions using the 
natural language in a meaningful manner. In social media a wide range of emotions are expressed 
implicitly or explicitly by using a mixture of words, emoticons and emojis. In this work we 
undertook a multi-emotion analysis for a tweet dataset sourced from CrowdFlower[1]. The dataset contains 40,000 instances of emotional text content in across 13 
emotions (such as 
happiness, sadness, and anger). We implemented emotion classifiers 
using the fine-tuning paradigm from pretrained Bidirectional Encoder Representations (BERT), a 
distilled version or BERT (DistilBERT) and Robustly Optimized approach (RoBERTa) base models. In 
total, we built an experimental setup of 24 models decribed in section 2. 

<h2><div>2. Methods</div></h2>

<h3><div>2.1 Data processing and feature selection </div></h3>

The tweets were processed to convert any emoticons and emojis into text. Table 1 
shows the original class distribution in the datset before data processing and feature enginering. 

<div style="margin-left: auto;
            margin-right: auto;
            width: 80%">

**Table 1.** Emotions distribution in dataset.

| Emotion    | Instances | 
|------------|-----------|
| neutral    | 8638      | 
| worry      | 8459      | 
| happiness  | 5209      | 
| sadness    | 5165      | 
| love       | 3842      | 
| surprise   | 2187      | 
| fun        | 1776      |
| relief     | 1526      | 
| hate       | 1323      | 
| empty      | 827       | 
| enthusiasm | 759       |
| boredom    | 179       |
| anger      | 110       |

</div>

We noticed that emoticons appeared in tweets with different labels. For instance a 
happy face smiley was present in tweets labelled as neutral, worry, 
sadness, empty, happiness, love, surprise, hate, boredom, relief, enthusiasm, and fun. 
To address this ambiguity we computed the similarity amongst emotion was by using 
pretrained word vectors from two models 'word2vec-google-news-300'. The emotions with the highest 
similarity were groupped except if the emotions had opposite meaning i.e., love and hate. In addition, we dropped the 'worry' instaces since they 
appeart to 
be related to all the other emotions. The final dataset contained in total 31,541 tweet of 
emotial content (Fig 1). 

<p
<img src='/plots/dist_1.png' height='200' title='dist1' alt='distribution1'/>  <img src='/plots/dist_emot.png' height='200' title='dist2' alt='distribution'/> </p>

Fig.2 (a) Initial emotion distribution in the dataset. (b) Emotions distribution aftercomputation of similarity and aggregation of emotions.



<h3><div>2.2 Classifiers </div></h3>

We built the classifiers by using the base models of BERT, DistilBERT and RoBERTa and 
a two dense layer classification head with 768 and 512 hidden units.

<h3><div>2.3 Experimental setup</div></h3>

The experimental setup for this work is described in Table 1. The experiments models
were designed by combining four factors: deep neural network architecture (classifier), the loss 
function (objective), learning rate and learning policy (scheduler). A total of 24 for experiments were built and evaluated (3×2×2×2).


**Table 2.**
Experimental setup with a total of 24 experiments. CE and wCE are the cross entropy 
and weigheted cross entropy loss functions to be minimised.


<div style="margin-left: auto;
            margin-right: auto;
            width: 80%">

| Experiment |  Network   | Loss | Learning<br />rate | Learning<br/>policy |
|------------|:----------:|:----:|:------------------:|:-------------------:|
| 01         |    BERT    |  CE  |       1e-05        |     one\_cycle      |
| 02         | DistilBERT |  CE  |       1e-05        |     one\_cycle      |
| 03         |  RoBERTa   |  CE  |       1e-05        |     one\_cycle      |
| 04         |    BERT    |  CE  |       1e-05        |       linear        |
| 05         | DistilBERT |  CE  |       1e-05        |       linear        |
| 06         |  RoBERTa   |  CE  |       1e-05        |       linear        |
| 07         |    BERT    |  CE  |       8e-06        |     one\_cycle      |
| 08         | DistilBERT |  CE  |       8e-06        |     one\_cycle      |
| 09         |  RoBERTa   |  CE  |       8e-06        |     one\_cycle      |
| 10         |    BERT    |  CE  |       8e-06        |       linear        |
| 11         | DistilBERT |  CE  |       8e-06        |       linear        |
| 12         |  RoBERTa   |  CE  |       8e-06        |       linear        |
| 13         |    BERT    | wCE  |       1e-05        |     one\_cycle      |
| 14         | DistilBERT | wCE  |       1e-05        |     one\_cycle      |
| 15         |  RoBERTa   | wCE  |       1e-05        |     one\_cycle      |
| 16         |    BERT    | wCE  |       1e-05        |       linear        |
| 17         | DistilBERT | wCE  |       1e-05        |       linear        |
| 18         |  RoBERTa   | wCE  |       1e-05        |       linear        |
| 19         |    BERT    | wCE  |       8e-06        |     one\_cycle      |
| 20         | DistilBERT | wCE  |       8e-06        |     one\_cycle      |
| 21         |  RoBERTa   | wCE  |       8e-06        |     one\_cycle      |
| 22         |    BERT    | wCE  |       8e-06        |       linear        |
| 23         | DistilBERT | wCE  |       8e-06        |       linear        |
| 24         |  RoBERTa   | wCE  |       8e-06        |       linear        | 

</div>

<h3><div>2.4 Training, validation and test approch </div></h3>

The dataset was split by label stratification in three subset with proportion of 80:10:10 for 
training, validation and test respectively. Table 3 shows the fixed hyperparameters used to fine 
tune all models.

**Table 3.**
Fixed experimental hyperparameters to fine-tune all experiments (classifiers).
<div style="margin-left: auto;
            margin-right: auto;
            width: 80%">

| Batch<br/>size | Hidden <br/>units | Dropout | Optimizer  |   Decay    | Epochs  |
|----------------|:-----------------:|:-------:|:----------:|:----------:|:-------:|
| 16             |    [768, 512]     |   0.3   |   AdamW    |    0.01    |    6    |

</div>

<h3><div>2.4 Evaluation </div></h3>

The models were evaluated on the test dataset with six metrics: accuracy, balanced accuracy (BA),
F1 score, recall precision and Matthew's correlation coefficient (MCC). In addition, the 
epochs required to achieve the best performance were logged. The performance of each model per 
metric was further ranked, followed by omnibus Friedman test and McNemar 
pair-wise comparison with uncertainty of 0.05 ($\alpha=0.05$). 

All code was written in python usin PyTorch framework and HuggingFace, mxlextend, sckitlearn, 
gensim, torchmetrics, seaborn, matplotlib, SciPy and scikit-posthocs libraries.

<h2><div>3. Results </div></h2>

Tables 4 shows the performace of the experiments in each evaluation metric for the
classification of emotional content in tweets. 

**Table 4.** Performance of each classifier measured in six metrics and training required to 
achieve the best performance. BA Balanced accuracy and MCC Mathew's correlation coefficient. Epoch denotes the number of 
epochs required to achieve the maximum validation accuracy

<div style="margin-left: auto;
            margin-right: auto;
            width: 90%">

| Exp |   Acc    |   BA    |   F1    |   Rec    |   Prec   |   MCC   | Epoch  |
|:----|:--------:|:-------:|:-------:|:--------:|:--------:|:-------:|:------:|
| 1   | 0\.6415  | 0\.5481 | 0\.5581 | 0\.5481  | 0\.5719  | 0\.4376 |   2    |
| 2   | 0\.6209  | 0\.5121 | 0\.5265 | 0\.5121  | 0\.5697  | 0\.4065 |   3    |
| 3   | 0\.6352  | 0\.5412 | 0\.5503 | 0\.5412  | 0\.5739  |  0\.43  |   3    |
| 4   | 0\.6469  | 0\.5562 | 0\.5633 | 0\.5562  | 0\.5741  | 0\.449  |   2    |
| 5   | 0\.6197  | 0\.5064 | 0\.5223 | 0\.5064  | 0\.5514  | 0\.3977 |   1    |
| 6   | 0\.6539  | 0\.5546 | 0\.5676 | 0\.5546  | 0\.5876  | 0\.4553 |   2    |
| 7   | 0\.6298  | 0\.5471 | 0\.543  | 0\.5471  | 0\.5452  | 0\.4245 |   2    |
| 8   | 0\.6222  | 0\.523  | 0\.5367 |  0\.523  | 0\.5573  | 0\.4048 |   2    |
| 9   | 0\.6339  | 0\.5023 | 0\.5221 | 0\.5023  | 0\.5823  | 0\.4163 |   1    |
| 10  | 0\.6453  | 0\.5523 | 0\.563  | 0\.5523  | 0\.5805  | 0\.4414 |   2    |
| 11  | 0\.6197  | 0\.5157 | 0\.5269 | 0\.5157  | 0\.5483  | 0\.4036 |   2    |
| 12  | 0\.6453  | 0\.5523 | 0\.563  | 0\.5523  | 0\.5805  | 0\.4414 |   2    |
| 13  |  0\.601  | 0\.5719 | 0\.5326 | 0\.5719  | 0\.5208  | 0\.4046 |   1    |
| 14  | 0\.5902  | 0\.5643 | 0\.5247 | 0\.5643  | 0\.5094  | 0\.3955 |   4    |
| 15  |  0\.601  | 0\.5719 | 0\.5326 | 0\.5719  | 0\.5208  | 0\.4046 |   1    |
| 16  | 0\.6022  | 0\.5793 | 0\.5318 | 0\.5793  | 0\.5182  | 0\.4155 |   4    |
| 17  | 0\.5962  | 0\.5631 | 0\.5254 | 0\.5631  | 0\.5141  | 0\.4011 |   2    |
| 18  | 0\.6022  | 0\.5793 | 0\.5318 | 0\.5793  | 0\.5182  | 0\.4155 |   4    |
| 19  | 0\.6067  | 0\.5971 | 0\.5401 | 0\.5971  | 0\.5249  | 0\.4233 |   4    |
| 20  | 0\.5959  | 0\.5554 | 0\.5299 | 0\.5554  | 0\.5159  | 0\.3947 |   1    |
| 21  | 0\.6035  | 0\.5851 | 0\.5349 | 0\.5851  | 0\.5204  | 0\.417  |   4    |
| 22  | 0\.6184  | 0\.5891 | 0\.549  | 0\.5891  | 0\.5343  | 0\.4277 |   1    |
| 23  | 0\.5927  | 0\.5636 | 0\.514  | 0\.5636  |  0\.505  | 0\.395  |   2    |
| 24  |  0\.62   | 0\.5919 | 0\.5478 | 0\.5919  | 0\.5348  | 0\.431  |   1    |
</div>

<h3><div>3.2 Ranking </div></h3>

Tables 5 shows the ranking of the experiments in each evaluation metric for the
classification of emotional content in tweets. 

**Table 5.** Ranking of 7 metrics to evaluate the performance of each classifier. 
BA Balanced accuracy and Mathew's correlation coefficient (MCC).

<div style="margin-left: auto;
            margin-right: auto;
            width: 100%">

| Exp |  Network   | Loss |    lr | scheduler  |   Acc    |    BA    |    F1    |   Rec    |   Prec   |   MCC    |  Epoch   |
|:----|:----------:|:----:|------:|:----------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| 1   |    BERT    |  CE  | 1e-05 | one\_cycle |   5\.0   |  17\.0   |   5\.0   |  17\.0   |   7\.0   |   5\.0   |  12\.5   |
| 2   | DistilBERT |  CE  | 1e-05 | one\_cycle |  10\.0   |  22\.0   |  19\.0   |  22\.0   |   8\.0   |  15\.0   |  18\.5   |
| 3   |  RoBERTa   |  CE  | 1e-05 | one\_cycle |   6\.0   |  19\.0   |   6\.0   |  19\.0   |   6\.0   |   7\.0   |  18\.5   |
| 4   |    BERT    |  CE  | 1e-05 |   linear   | **2\.0** |  12\.0   | **2\.0** |  12\.0   |   5\.0   | **2\.0** |  12\.5   |
| 5   | DistilBERT |  CE  | 1e-05 |   linear   |  12\.5   |  23\.0   |  22\.0   |  23\.0   |  10\.0   |  21\.0   | **4\.0** |
| 6   |  RoBERTa   |  CE  | 1e-05 |   linear   | **1\.0** |  14\.0   | **1\.0** |  14\.0   | **1\.0** | **1\.0** |  12\.5   |
| 7   |    BERT    |  CE  | 8e-06 | one\_cycle |   8\.0   |  18\.0   |   9\.0   |  18\.0   |  12\.0   |   9\.0   |  12\.5   |
| 8   | DistilBERT |  CE  | 8e-06 | one\_cycle |   9\.0   |  20\.0   |  11\.0   |  20\.0   |   9\.0   |  16\.0   |  12\.5   |
| 9   |  RoBERTa   |  CE  | 8e-06 | one\_cycle |   7\.0   |  24\.0   |  23\.0   |  24\.0   | **2\.0** |  12\.0   | **4\.0** |
| 10  |    BERT    |  CE  | 8e-06 |   linear   |   3\.5   |  15\.5   |   3\.5   |  15\.5   |   3\.5   |   3\.5   |  12\.5   |
| 11  | DistilBERT |  CE  | 8e-06 |   linear   |  12\.5   |  21\.0   |  18\.0   |  21\.0   |  11\.0   |  19\.0   |  12\.5   |
| 12  |  RoBERTa   |  CE  | 8e-06 |   linear   |   3\.5   |  15\.5   |   3\.5   |  15\.5   |   3\.5   |   3\.5   |  12\.5   |
| 13  |    BERT    | wCE  | 1e-05 | one\_cycle |  19\.5   |   7\.5   |  13\.5   |   7\.5   |  16\.5   |  17\.5   | **4\.0** |
| 14  | DistilBERT | wCE  | 1e-05 | one\_cycle |  24\.0   |   9\.0   |  21\.0   |   9\.0   |  23\.0   |  22\.0   |  22\.0   |
| 15  |  RoBERTa   | wCE  | 1e-05 | one\_cycle |  19\.5   |   7\.5   |  13\.5   |   7\.5   |  16\.5   |  17\.5   | **4\.0** |
| 16  |    BERT    | wCE  | 1e-05 |   linear   |  17\.5   |   5\.5   |  15\.5   |   5\.5   |  19\.5   |  13\.5   |  22\.0   |
| 17  | DistilBERT | wCE  | 1e-05 |   linear   |  21\.0   |  11\.0   |  20\.0   |  11\.0   |  22\.0   |  20\.0   |  12\.5   |
| 18  |  RoBERTa   | wCE  | 1e-05 |   linear   |  17\.5   |   5\.5   |  15\.5   |   5\.5   |  19\.5   |  13\.5   |  22\.0   |
| 19  |    BERT    | wCE  | 8e-06 | one\_cycle |  15\.0   | **1\.0** |  10\.0   | **1\.0** |  15\.0   |  10\.0   |  22\.0   |
| 20  | DistilBERT | wCE  | 8e-06 | one\_cycle |  22\.0   |  13\.0   |  17\.0   |  13\.0   |  21\.0   |  24\.0   | **4\.0** |
| 21  |  RoBERTa   | wCE  | 8e-06 | one\_cycle |  16\.0   |   4\.0   |  12\.0   |   4\.0   |  18\.0   |  11\.0   |  22\.0   |
| 22  |    BERT    | wCE  | 8e-06 |   linear   |  14\.0   |   3\.0   |   7\.0   |   3\.0   |  14\.0   |   8\.0   | **4\.0** |
| 23  | DistilBERT | wCE  | 8e-06 |   linear   |  23\.0   |  10\.0   |  24\.0   |  10\.0   |  24\.0   |  23\.0   |  12\.5   |
| 24  |  RoBERTa   | wCE  | 8e-06 |   linear   |  11\.0   | **2\.0** |   8\.0   | **2\.0** |  13\.0   |   6\.0   | **4\.0** |



</div>

<h3><div>3.4 Statistical comparison </div></h3>

<p>After applying the Friedman test to the prediction, we rejected the null hypothesis that the data came from the same distribution.
Then we compared the models using the McNemar pairwise. The comparison is shown in figure 2.</P>


<p align="center">
<img src='/plots/mcnemar.png' height='350'>
</p>

<p align="center">
Figure 2. McNemar pairwise experiments performance comparison with $\alpha = 0.05$.</p>

<p>
The highes metrics values were obtained by experiments Ex-06, followed by Exp-04. Howerver, the 
McNemar test shows with 95 % certainty that there is no significant difference in the classification
performance of Exp-06, Exp-04, Exp-10 and Exp-12. These top classifiers were built on BERT or RoBERTa, with 
learning rates of $1\times 10^{-5}$ and $8\times 10^{-6}$. They also followed a linear police and learnt to minimise the CE loss fuction.
From all metrics used to evaluate the models' performace we considered the MCC to be the most representative of the performance of the classifiers since it takes into consideration the total measurement of imbalance of the dataset. Furthermore, we noted that most of the models required less than four epochs to reach their best validation performance.</p>
            



<h2><div>4. Discussion </div></h2>

In spite of the limitations of the dataset we obtained accuraccy of 65.4% and MCC of 45.5% in 
the MCC. We identified the best classifiers by using non-parametric statistics and further 
posthoc comparison of each model. In this work we attempted to maintain as much emotional 
content as possible for the classifier to be able to learn and deal with ambiguity. We therefore 
used pretrained word vectors to aggregate emotions with high similarity amongst them.  Work 
conducted by [2,3] suggest that emotions as worry and neutral require their own individual 
study. The quality of the labelling of the dataset and class imbalance together with 
the mismatch of emoticon have also an impact on the models' performance. Our dataset is 
dated in 2016 (available to the public) when most of the emoticons were manually input rather than 
automatically predicted by the text input. A further recomendation is to use ourmodels with more 
recent dataset to be able to identyfing potential flaws and opporrunities to improve the models' 
performance. In summary the main challeges of this work were the class imbalance, labels and 
mismatch emoticons. 

<h3><div> References </div></h3>

1. https://query.data.world/s/m3dkicscou2wd5p2d2ejd7ivfkipsg
2. [Does Neutral Affect Exist? How Challenging Three Beliefs About Neutral Affect Can Advance 
   Affective Research](https://pubmed.ncbi.nlm.nih.gov/31787911/) (Gasper, Karen et al., Front 
   Psychol. 2019  )
3. [Identifying Worry in Twitter: Beyond Emotion Analysis](https://aclanthology.org/2020.nlpcss-1.9) (Verma et al., NLP+CSS 2020)
