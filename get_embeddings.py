import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

import nltk
import numpy as np
import pickle
import gc
import re
import pandas as pd
import argparse


def remove_url(text, replace_token):
    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(regex, replace_token, text)

def remove_mentions(text, replace_token):
    return re.sub(r'(?:@[\w_]+)', replace_token, text)


def remove_hashtags(text, replace_token):
    return re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", replace_token, text)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_shifts(input_path):
    shifts_dict = {}
    df_shifts = pd.read_csv(input_path, sep=',', encoding='utf8')
    for idx, row in df_shifts.iterrows():
        shifts_dict[row['word']] = row['shift_index']
    return shifts_dict

def tokens_to_batches(ds, tokenizer, batch_size, max_length):

    batches = []
    batch = []
    batch_counter = 0
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    print('Dataset: ', ds)
    counter = 0
    with open(ds, 'r', encoding='utf8') as f:

        for line in f:
            counter += 1

            if counter % 1000 == 0:
                print('Num articles: ', counter)


            content = re.search("'body': (.*), 'parent_id':", line)

            if content:
                content = content.groups()[0]
            else:
                content = ''
            title = re.search("'title': (.*), 'subreddit':", line)
            if title:
                title = title.groups()[0]
            else:
                title = ''

            text = title + ' ' + content

            text = remove_url(text, '')

            text = text.replace("\d", "'d")
            text = text.replace("\\t", "'t")
            text = text.replace("\s", "'s")
            text = text.replace("\ll", "'ll")
            text = text.replace("\m", "'m")
            text = text.replace("\\ve", "'ve")
            text = text.replace("[deleted]", "")
            text = " ".join(text.split())

            sents = []

            for sent in sent_tokenizer.tokenize(text):
                sent = sent.strip().lower()

                marked_sent = "[CLS] " + sent + " [SEP]"
                sents.append(marked_sent)


            marked_text = " ".join(sents)

            tokenized_text = tokenizer.tokenize(marked_text)

            for i in range(0, len(tokenized_text), max_length):

                batch_counter += 1
                input_sequence = tokenized_text[i:i + max_length]

                indexed_tokens = tokenizer.convert_tokens_to_ids(input_sequence)

                batch.append((indexed_tokens, input_sequence))

                if batch_counter % batch_size == 0:
                    batches.append(batch)
                    batch = []

    print()
    print('Tokenization done!')
    print('len batches: ', len(batches))

    return batches


def get_token_embeddings(batches, model, batch_size):

    token_embeddings = []
    tokenized_text = []
    counter = 0

    for batch in batches:
        counter += 1
        if counter % 1000 == 0:
            print('Generating embedding for batch: ', counter)
        lens = [len(x[0]) for x in batch]
        max_len = max(lens)
        tokens_tensor = torch.zeros(batch_size, max_len, dtype=torch.long).cuda()
        segments_tensors = torch.ones(batch_size, max_len, dtype=torch.long).cuda()
        batch_idx = [x[0] for x in batch]
        batch_tokens = [x[1] for x in batch]

        for i in range(batch_size):
            length = len(batch_idx[i])
            for j in range(max_len):
                if j < length:
                    tokens_tensor[i][j] = batch_idx[i][j]

        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)


        for batch_i in range(batch_size):

            # For each token in the sentence...
            for token_i in range(len(batch_tokens[batch_i])):

                # Holds 12 layers of hidden states for each token
                hidden_layers = []

                # For each of the 12 layers...
                for layer_i in range(len(encoded_layers)):
                    # Lookup the vector for `token_i` in `layer_i`
                    vec = encoded_layers[layer_i][batch_i][token_i]

                    hidden_layers.append(vec)

                hidden_layers = torch.sum(torch.stack(hidden_layers)[-4:], 0).reshape(1, -1).detach().cpu().numpy()

                token_embeddings.append(hidden_layers)
                tokenized_text.append(batch_tokens[batch_i][token_i])

    return token_embeddings, tokenized_text


def average_save_and_print(vocab_vectors, save_path):
    for k, v in vocab_vectors.items():
        #v = np.array(v)
        #median = np.median(v, axis=0)
        #vocab_vectors[k] = median

        if len(v) == 2:
            avg = v[0] / v[1]
            vocab_vectors[k] = avg

    with open(save_path, 'wb') as handle:
        pickle.dump(vocab_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)




def get_time_embeddings(embeddings_path, datasets, tokenizer, model, batch_size, max_length):

    vocab_vectors = {}
    vocab_vectors_avg = {}

    for ds in datasets:

        year = ds[-8:-4]

        all_batches = tokens_to_batches(ds, tokenizer, batch_size, max_length)
        chunked_batches = chunks(all_batches, 1000)
        num_chunk = 0

        for batches in chunked_batches:
            num_chunk += 1
            print('Chunk ', num_chunk)

            token_embeddings, tokenized_text = get_token_embeddings(batches, model, batch_size)

            splitted_tokens = []
            splitted_array = np.zeros((1, 768))
            prev_token = ""
            prev_array = np.zeros((1, 768))

            for i, token_i in enumerate(tokenized_text):

                array = token_embeddings[i]

                if token_i.startswith('##'):

                    if prev_token:
                        splitted_tokens.append(prev_token)
                        prev_token = ""
                        splitted_array = prev_array

                    splitted_tokens.append(token_i)
                    splitted_array += array

                else:

                    if token_i + '_' + year in vocab_vectors:
                        vocab_vectors[token_i + '_' + year][0] += array
                        vocab_vectors[token_i + '_' + year][1] += 1
                    else:
                        vocab_vectors[token_i + '_' + year] = [array, 1]

                    if splitted_tokens:
                        sarray = splitted_array / len(splitted_tokens)
                        stoken_i = "".join(splitted_tokens).replace('##', '')


                        if stoken_i + '_' + year in vocab_vectors:
                            vocab_vectors[stoken_i + '_' + year][0] += sarray
                            vocab_vectors[stoken_i + '_' + year][1] += 1
                        else:
                            vocab_vectors[stoken_i + '_' + year] = [sarray, 1]

                        splitted_tokens = []
                        splitted_array = np.zeros((1, 768))

                    prev_array = array
                    prev_token = token_i

            del token_embeddings
            del tokenized_text
            del batches
            gc.collect()

        print('Sentence embeddings generated.')

    print("Length of vocab after training: ", len(vocab_vectors.items()))

    average_save_and_print(vocab_vectors, embeddings_path)
    del vocab_vectors
    del vocab_vectors_avg
    gc.collect()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument('--path_to_model', type=str,
                        help='Paths to the fine-tuned BERT model',
                        default='models/model_liverpool/pytorch_model.bin')
    parser.add_argument('--path_to_datasets', type=str,
                        help='Paths to each of the time period specific corpus separated by ;',
                        default='data/liverpool/LiverpoolFC_2013.txt;data/liverpool/LiverpoolFC_2017.txt',)
    parser.add_argument('--embeddings_path', type=str,
                        help='Path to output time embeddings',
                        default='embeddings/liverpool.pickle')
    parser.add_argument('--shifts_path', type=str,
                        help='Path to gold standard semantic shifts path',
                        default='data/liverpool/liverpool_shift.csv')
    args = parser.parse_args()

    datasets = args.path_to_datasets.split(';')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    state_dict = torch.load(args.path_to_model)
    model = BertModel.from_pretrained('bert-base-uncased', state_dict=state_dict)

    model.cuda()
    model.eval()

    embeddings_path = args.embeddings_path

    shifts_dict = get_shifts(args.shifts_path)

    get_time_embeddings(embeddings_path, datasets, tokenizer, model, args.batch_size, args.max_length)

