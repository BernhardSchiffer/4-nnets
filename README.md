# Neural Networks

Before you get started, please head over to Moodle and download the `theses.tsv` (tab separated values) data set, that contains about 3000 thesis titles along with their type (diploma, bachelor, master) and category (internal/external).
Here is an example:

```
1995	extern	Diplom	Analyse und Leistungsvergleich von zwei Echtzeitsystemen für eingebettete Anwendungen
1995	intern	Diplom	Erfassung und automatische Zuordnung von Auftragsdaten für ein Dienstleistungsunternehmen mit Hilfe von Standardsoftware - Konzeption und Realisierung
```

As you can see, the format is

```
date<tab>{intern,extern}<tab>{Diplom,Bachelor,Master}<tab>Title...
```


## Skip-grams

[Tomas Mikolov's original paper](https://arxiv.org/abs/1301.3781) for word2vec is not very specific on how to actually compute the embedding matrices.
Xin Ron provides a [much more detailed walk-through of the math](https://arxiv.org/pdf/1411.2738.pdf), I recommend you go through it before you continue with this assignment.

Now, while the original implementation was in C and estimates the matrices directly, in this assignment, we want to use pytorch (and autograd) to train the matrices.
There are plenty of example implementations and blog posts out there that show how to do it, I particularly recommend [Mateusz Bednarski's](https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb) version.

1. Familiarize yourself with skip-grams and how to train them using pytorch.
2. Use the titles from `theses.tsv` to compute word embeddings over a context of 5. Note: it may be helpful to lower-case the data.
3. Analyze: What are the most similar words to "Konzeption", "Cloud" and "virtuelle"
4. Play: Using the computed embeddings: can you identify the most similar theses?


## RNN-LM

Implement a basic (word-based) [RNN](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)-LM for the theses titles.
You can use either the embeddings from above or learn a dedicated [embedding layer](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html).

1. Implement, evaluate: Using 5-fold cross-validation, what is the average perpexity?
	- recall [assignment 2](https://github.com/seqlrn/2-markov-chains): what perplexity does a regular 4-gram have on the same splits?
2. Sample a few random theses titles from the RNN-LM
	- are these any good/better than from [assignment 2](https://github.com/seqlrn/2-markov-chains)?


## RNN for classification

The `theses.tsv` also contains type (diploma, bachelor, master) and category (internal/external) for each thesis.
In this part, we want to classify whether the thesis is bachelor or master; and if it's internal or external.
Since pytorch provides most things sort-of out of the box, compare the following on a 5-fold x/validation: (vanilla) RNN, GRU, LSTM, bi-LSTM; which activations did you use?

1. Filter out all diploma theses; they might be too easy to spot because they only cover "old" topics.
2. Train and evaluate your models on a 5-fold cross-validation; as in RNN-LM, you can either learn the embeddings or re-use the ones from the skip-gram.
3. Assemble a table: Recall/precision/F1 measure for each of above listed recurrent model architectures. Which one works best?
4. Bonus: Apply your best classifier to the remaining diploma theses; are those on average more bachelor or master? :-)
