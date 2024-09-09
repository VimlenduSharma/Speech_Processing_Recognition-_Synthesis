Speech Recognition with Wav2Vec2(PyTorch).
Speech recognition using pre-trained models from wav2vec.
The process of Speech Recognition takes place as:-
Extract the acoustic features from audio waveform.
Estimate the class of the acoustic features frame-by-frame.
Generate Hypothesis from sequence of class probabilities.
Torch audio provides easy access to pre-trained weights and associated information, such as expected sample rate and class labels.
First we will create a Wav2Vec2 model that performs the feature extraction and classification.
There are two types of Wav2Vec2 pre-trained weights available in torch audio. The ones fine-tuned for ASR task, and the ones not fine-tuned.
Wav2Vec2(HuBERT) models are trained in self-supervised manner. First trained with audio only for representation learning, then fine-tuned for specific tasks with additional labels.
Wav2Vec2 models fine-tuned for ASR(Automatic Speed Recognition) task can perform feature extraction and classification with one step.
From the seq. of label probabilities , we want to generate transcripts. The process of generate hypothesis is called “decoding”.
Decoding is more elaborate than simple classification because decoding at certain time step can be affected by surrounding observations.
The ASR model is fine-tuned using a loss function called Connectionist Temporal Classification (CTC), In CTC a blank token (ϵ) is a special token which represents a repetition of the previous symbol. In decoding, these are simply ignored.
