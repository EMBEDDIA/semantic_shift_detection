# Code for experiments conducted in the paper 'Leveraging Contextual Embeddings for Detecting Diachronic Semantic Shift' published in proceedings of LREC 2020 conference #

Please cite the following paper [[bib](https://gitlab.com/matej.martinc/semantic_shift_detection/-/blob/master/bibtex.js)] if you use this code:

Matej Martinc, Petra Kralj Novak and Senja Pollak. Leveraging Contextual Embeddings for Detecting Diachronic Semantic Shift. In Proceedings of the Twelfth International Conference on Language Resources and Evaluation (LREC 2020). Marseille, France.


## Installation, documentation ##

Published results were produced in Python 3 programming environment on Linux Mint 18 Cinnamon operating system. Instructions for installation assume the usage of PyPI package manager.<br/>
Clone the project from the repository with 'git clone https://gitlab.com/matej.martinc/semantic_shift_detection'<br/>
Install dependencies if needed: pip install -r requirements.txt

### To reproduce the results on the LiverpoolFC corpus published in the paper run the code in the command line using following commands: ###

Generate time specific representation for each word using the already fine-tuned model:<br/>
python get_embeddings.py

Visualize everything and calculate Pearson correlation:<br/>
python visualize.py

To fine-tune custom BERT model on the corpus<br/>
python fine_tuning.py


## Contributors to the code ##

Matej Martinc<br/>

* [Knowledge Technologies Department](http://kt.ijs.si), Jožef Stefan Institute, Ljubljana
