### addition_rnn
- sequence2sequence模型
- 序列长度是不变的
- 无需加载数据集，对于学习序列预测很有帮助
- 最后的准确率也是相当之高
### babi_rnn
问答系统：
1. 给出一个简单的故事
2. 提出一个简单的问题
3. 预测一个简单的答案
### babi_rnn_functions
用于**babi_rnn.py**的调试
### lstm_seq2seq
英语翻译成法语
Here's a summary of our process:
1. Turn the sentences into 3 Numpy arrays, `encoder_input_data, decoder_input_data, decoder_target_data`:
    - `encoder_input_data` is a 3D array of shape (`num_pairs, max_english_sentence_length, num_english_characters`) containing a one-hot vectorization of the English sentences.
    - `decoder_input_data` is a 3D array of shape (`num_pairs, max_french_sentence_length, num_french_characters`) containg a one-hot vectorization of the French sentences.
    - `decoder_target_data` is the same as `decoder_input_data` but *offset by one timestep*. `decoder_target_data[:, t, :]` will be the same as `decoder_input_data[:, t + 1, :]`
2. Train a basic LSTM-based Seq2Seq model to predict `decoder_target_data` given `encoder_input_data` and `decoder_input_data`. Our model uses teacher forcing.
3. Decode some sentences to check that the model is working (i.e. turn samples from `encoder_input_data` into corresponding samples from `decode_target_data`)

### mnist_hierarchical_rnn
使用分层(hierarchical)RNN对MNIST进行分类。

HRNN可以