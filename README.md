# CSCI567 Code-switched Text Generation Contents

#### SEAME.csv: The Mandarin-English Code-Switching in South-East Asia (SEAME) dataset, mix of code-switched data and monolingual data
#### MONO.csv: Monolingual sentences from the SEAME dataset, collection of both English and Mandarin instances
#### CS_train.csv: 50K code-switched data. First column: original data entry; Second column: English translation; Third column: Mandarin translation
#### mBert_evaluation_for_future_work.ipynb : Code for performing cosine similarity analysis on the machine generated CS text against the reference text from the SEAME Corpus 
#### data.py: Data preparation script: extracting sentences, data translation, training and testing sets creation
#### vocab.py: Vocabulary extraction and word embeddings generation
#### model.py: Transformer model with training process
#### GBS.py: Grid Beam Search, operating on decoder's output to force generation of code-switched data
#### transformer_.py: Transformer based machine translation model (non-functional due to error while trianing)
#### trigram.py: Trigram model with KneserNey Smoothing, to be used to evaluate quality of code-switched data

#### Order of execution: data.py -> vocab.py -> transformer_.py
