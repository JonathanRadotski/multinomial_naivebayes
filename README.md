## Multinomial Naive Bayes

This little documentation is about implementation of **Multinomial Naive Bayes** to predict whether A Youtube comment (topic: "Political Year 2019 in Indonesia") is a spam or not. SPAM in this project have a definition of a unsoliticed advertising and out of the topic comments. Many platform have implemented these feature to reduce spamming activities in the internet.

This little project uses Multinomial Naive Bayes as its algorithm. Probabilistic model gives an insight about word behaviour. Naive Bayes itself calculate probabilities between labels and predict which label the test data goes to.

![Naive Bayes Theorem](http://uc-r.github.io/public/images/analytics/naive_bayes/naive_bayes_icon.png)

Why use Naive Bayes? It is simple and perform well in many real-world problems (such as: Document categorization, e-mail filtering, spam detection). Many complex algorithm outperform naive bayes such as Neural Network, Evolutionary Algorithm.

Multinomial naive bayes is a naive bayes based algorithm which represent frequencies of an events happening. This algorithm predict labels based on how many times a feature have occured.

![Multinomial Naive Bayes Theorem](https://wikimedia.org/api/rest_v1/media/math/render/svg/52bd0ca5938da89d7f9bf388dc7edcbd546c118e)

### Pseudo Code

First step is to split dataset into its own label. The main priorities is to group sentences for later process. Other priorities we need to do is calculate occurance of spam and not spam data, vocabulary, and total data. These elements is needed due to provide prior elements of naive bayes.

![Prior](https://cdn-images-1.medium.com/max/1600/1*-PkUQ4n42T1YLaK-jXI9VQ.png)

```markdown
# split_text_train(dataset):
##    for data in each row dataset:
      split data into words
      splitted.append(words)
      calculate occurance of spam (1) from splitted
      calculate occurance of not spam(0) from splitted
      insert data dictionary spam
      insert data dictionary not spam
##      if data not in data dictionary:
          vocabulary += 1
      sum data += 1
##    return splitted, data dictionary, occurance of spam, occurance of not spam, vocabulary, sum data
```

Second step is to provide words in each of the respective labels. These dictionaries provide information of each word that occured in each label. Why not do this process in previous method? Because we want to keep track of data without losing their track (Ex: each word from particular sentence)

```markdown
# count_class_freq(data dictionary):
##    for word in data dictionary[spam]:
##       if word not in word dictionary spam:
            add to word dict spam[spam]
            
##    for word in data dictionary[not spam]:
##       if word not in word dictionary not spam:
            add to word dict not spam[not spam]
##     return word dict spam, word dict not spam
```

Training time!!! :D we need to train the algorithm so they can predict accurately. In this method we want to calculate ***number of word*** occured in each label. We want to calculate the likelihood of each word given spesific class.
![Likelihood](https://cdn-images-1.medium.com/max/1600/1*179n7MYlMneiPvhhUizSIg.png)

Wait... there is a problem. What if a word that we test doesnt occured in the training data? will it destroy the algorithm by giving 0 to probability values? Of course. Based on that problem, we need to prevent the problem by inserting ***smoothing*** . In this project we add 1 for each data to calculate their likelihoods (This method is called ***laplace smoothing*** ). By adding 1 to each data, the data with 0 occurance in the dataset will give probability value and provide likelihoods for future process. 

![Likelihood with smoothing](https://cdn-images-1.medium.com/max/1600/1*PB2b5dB2rHffvx_g-wcSng.png)


```markdown
# training(dataset, test data, data dictionary, word dict spam, word dict not spam, vocabulary, alpha = 1):
  _alpha = laplace smoothing (prevents probabilities being 0)_

##    for word in dataset:
        check if word exist in word dict spam (yes = spam word[word] + 1 | no = spam word [word] +0)
        check if word exist in word dict not spam (yes = not spam word[word] + 1 | no = not spam word [word] +0)
        
##    for word in test data:
        check if word exist in word dict spam (yes = spam word[word] + 1 | no = spam word [word] +0)
        check if word exist in word dict not spam (yes = not spam word[word] + 1 | no = not spam word [word] +0)
        
##    for word in spam_words:
        prob_spam_words[word] = (spam words[word] + alpha)/(num of word in spam + vocabulary)

##    for word in not_spam_words:
        prob_not_spam_words[word] = (not spam words[word] + alpha)/(num of word in not spam + vocabulary)

##    return prob_spam_words, prob_not_spam_words
```

The last step is to predict the test set. We need to calculate combined probability value of each word that occured in the test set. From that, we multiply it by prior and divide it with combined probability value of each word that occured in the test set. But without dividing we get a derived value that can predict our test set being a spam set or a not spam set. Lastly, we need to calculate the max value between spam meter and not spam meter to predict which class our tes set belongs to.

![predict Mnb 1](https://cdn-images-1.medium.com/max/1600/1*nSjjuW2SBNcGZ3cN61E4NQ.png)

![predict Mnb 1](https://cdn-images-1.medium.com/max/1600/1*onKHO0N1AKcVoDyVnHE7CA.png)

```markdown
# predict(test data, prob_spam_words, prob_not_spam_words, occurance of spam, occurance of not spam, sum data):

##    for word in test data:
        check if word exist in word dict spam (yes = prob spam word[word] | else = prob spam word [word])
        check if word exist in word dict not spam (yes = prob not spam word[word] | else = prob not spam word [word])

##    for value in probability spam:
        spam meter = spam meter * value
        
      spam meter = spam meter * (occurance of spam/sum data)


##    for value in probability not spam:
        not spam meter = not spam meter * value
        
      not spam meter = not spam meter * (occurance of not spam/sum data)
      
      if spam meter > not spam meter= prediction(spam)
      if not spam meter > spam meter= prediction(not spam)
      
      prediction value = argmax(spam meter, not spam meter)

##    return prediction, prediiction value

```

### End words

Thanks for reading this little documentation. Hope this help y'all. Check out my other algorithm from scratch. Your input will help me code better. :D
