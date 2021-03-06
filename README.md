# OTDA
Online Topic Modeling for short texts


Large amount of short texts are generated on the social media across the internet. Discovering knowledge from them has attracted lot of attention in the recent past as topic discovery from these short texts can reflect hidden information. However, limited information is available in short texts as they are sparse and ambiguous. A semantics-assisted non-negative matrix factorization (SeaNMF)model is proposed in the recent past to discover topics from the short texts which incorporates word-context semantic relationship using skip-gram view of the corpus. We bulid upon this SeaNMF model and propose an online topic discovery algorithm (OTDA) for short texts. This OTDA works with one data point or onechunk of data points at a time instead of keeping the entire data in the memory.We consider a couple of public data sets an internal data to conduct experiment with OTDA on them using one-pass and multi-pass iterations of the algorithm.The results are promising


Data sets links used for experimentation:<br>
<ul>
  <li>Yahoo: https://webscope.sandbox.yahoo.com/catalog.php?datatype=l </li>
  <li>Stack: https://www.kaggle.com/c/predict-closed-questions-on-stack-overflow/download/train.zip </li>
  <li>Snippets: http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 </li>
  <li>Optum: Internal data</li>
</ul>


Online SeaNMF code is modified from https://github.com/tshi04/SeaNMF

LDA Baselines used:
1. Online LDA : scikit learn
2. DTM : Gensim library
3. Adaptive LDA: https://github.com/Wind-Ward/OnLine-LDA
