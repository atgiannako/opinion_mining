## Thesis Title: 
Using Relational Features to Detect Opinion Targets

## Abstract
Mining the opinions people express has applications in numerous fields, from customer relations to policy creation. While in certain cases knowing the overall sentiment in a document has value, in general that analysis level is not enough. A more informative view is provided by attaching opinions to specific targets. These more fine grained opinions, known as aspect-based opinions, require the extraction of the relevant aspects. For example in the phrase "This thesis will be awesome" the aspect is "thesis".

In my thesis, I create a method for extracting opinion targets by mixing word embeddings, vocabulary and syntax features. Tools such as SyntaxNet, Stanford Parser and Spacy will be used in extracting relational information in a sentence. Models such as Google's Word2Vec will be used for word embeddings. The resulting feature vectors will be used in a classification task to find opinion aspects (train classifier with publicly available corpora). The results obtained with each parser will be compared with a baseline model.

This Master thesis is done in the Swisscom Digital Lab in collaboration with the Laboratory of Artificial Intelligence at EPFL.
