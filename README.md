# Naive Bayes Text Classification

This is a simple Naive Bayes Classifier to classify biographies.

### Usage
    python3 textClassifier.py file n

        file: the file of the corpus
        n: the number of entries to use as the training set

### Input Format
- Separate biographies are separated by 1 or more blank lines
- The first line is the name of the person
- The second line is the category
- The remaining lines are the biography
- See [bioCorpus.txt](./bioCorpus.txt)

### Output Format
- For each person in the test set:
    - the probabilities associated with each category
    - the prediction
    - a statement whether this is right or wrong
- The overall accuracy of the classifier on the test set
- See [bioCorpus_output.txt](./bioCorpus_output.txt)


### Notes
- Please put [stopwords.txt](./stopwords.txt) in the same working
directory as the program
- The output will be saved to "{input_filename}_output.txt"