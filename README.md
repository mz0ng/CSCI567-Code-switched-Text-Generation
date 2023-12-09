# CSCI567 Code-switched Text Generation Contents

#### SEAME.csv: The Mandarin-English Code-Switching in South-East Asia (SEAME) dataset, mix of code-switched data and monolingual data
#### MONO.csv: Monolingual sentences from the SEAME dataset, collection of both English and Mandarin instances
#### CS_train.csv: 50K code-switched data. First column: original data entry; Second column: English translation; Third column: Mandarin translation
#### mBert_evaluation_for_future_work.ipynb : Code for performing cosine similarity analysis on the machine generated CS text against the reference text from the SEAME Corpus 

#### data.py: Data preparation script: extracting sentences, data translation, training and testing sets creation
#### vocab.py: Vocabulary extraction and word embeddings generation
#### model.py: Transformer model with training process
#### trigram.py: Trigram model with KneserNey Smoothing, to be used to evaluate quality of code-switched data
