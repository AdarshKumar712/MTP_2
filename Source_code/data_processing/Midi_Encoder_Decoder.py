import os
from miditok import REMIEncoding, get_midi_programs
from miditoolkit import MidiFile
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece, WordLevel
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer, WordLevelTrainer
from tqdm import tqdm
import numpy as np


def get_midi_tokenizer(midi_tok_type='REMI', args=None):
    if midi_tok_type=='REMI':
        if args!=None:
            tokenizer = REMIEncoding(args['pitch_range'], args['beat_res'], args['nb_velocities'], args['additional_tokens'])
        else:
            text = '[CLS]'
            pitch_range = range(21, 109)
            beat_res = {(0, 4): 8, (4, 12): 4}
            nb_velocities = 32
            additional_tokens = {'Chord': False, 'Rest': True, 'Tempo': False, 'Program': False,
                                 'rest_range': (2, 8),  # (half, 8 beats)
                                 'nb_tempos': 32,  # nb of tempo bins
                                 'tempo_range': (40, 250)}  # (min, max)
            tokenizer = REMIEncoding(pitch_range, beat_res, nb_velocities, additional_tokens)
    else:
        raise NotImplementedError
    return tokenizer
    
# encoded.txt expects tokens to be encoded with '[SEP]' as separator
def train_tokenizer(vocab_size=300, alg='BPE', encoded_file='encoded.txt', sep_token='[SEP]', unk_token='[UNK]', cls_token='[CLS]', pad_token='[PAD]'):
    with open("encoded.txt",'r',encoding="utf-8") as f:
        text = f.read()
    spl_tokens = [unk_token, cls_token, sep_token, pad_token]
    if alg == 'BPE':
        tokenizer = Tokenizer(BPE(unk_token = unk_token))
        trainer = BpeTrainer(vocab_size=vocab_size, show_progress=True, special_tokens = spl_tokens)
    elif alg == 'UNI':
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(vocab_size=vocab_size, show_progress=True, unk_token= unk_token, special_tokens = spl_tokens)
    elif alg == 'WPC':
        tokenizer = Tokenizer(WordPiece(unk_token = unk_token))
        tokenizer.model.max_input_chars_per_word = 50000
        trainer = WordPieceTrainer(vocab_size=vocab_size, show_progress=True, special_tokens = spl_tokens)
    else:
        tokenizer = Tokenizer(WordLevel(unk_token = unk_token))
        trainer = WordLevelTrainer(vocab_size=vocab_size, show_progress=True, special_tokens = spl_tokens)
    
    tokens_list = text.split(sep_token)
    tokenizer.train_from_iterator(tokens_list, trainer)
    return tokenizer

def load_pretrained_tokenizer(file_path='data/vocab.json'):
    return Tokenizer.from_file(file_path)

def preprocess_tokens(tokenizer_midi, midi_dir="./src_files", sep_token='[SEP]', unk_token='[UNK]', cls_token='[CLS]', pad_token='[PAD]'):
    
    files = os.listdir("session")
    
    tokens_list = []
    for i in tqdm(files):
        if i.endswith(".mid"):
            midi = MidiFile("session/"+i)
            if len(tokenizer_midi.midi_to_tokens(midi))==0:
                continue
            tokens_list.append(tokenizer_midi.midi_to_tokens(midi)[0])
    np.save("./data/train_base.npy", np.array(tokens_list))
    start = int(0x2200)
    text_list = []
    for tokens in tqdm(tokens_list):
        text_1 = ''
        for i in range(len(tokens)):
            number = start+tokens[i]
            text_1+= chr(number)
        text.append(text_1)
     
    return text
    
def decode_tokens(tokens_list, tokenizer_midi, tokenizer_nlp, alg='BPE', save_path='./Results/', file_prefix=''):
    tokens_before_midi = []
    
    for i in tokens_list:
        s1 = tokenizer_nlp.decode(tokens)
        if alg!='WPC':
            s2 = "".join(s1.split(' '))
        else:
            s2 = "".join(s1.split(' ##'))
        tokens_real = []
        for i in range(len(s2)):
            tokens_real.append(int(ord(s2[i])-int(0x2200)))
        
        tokens_before_midi.append(tokens_real)
    
    for i in range(len(tokens_before_midi)):
        save_name = save_path + file_prefix + str(i) + '.mid'
        generated_midi = tokenizer_nlp.tokens_to_midi([tokens_before_midi[i]], [(0, False)])
        generated_midi.dump(save_name)
    
    
if __name__=='__main__':    
    
    tokenizer_midi = get_midi_tokenizer()
    print("Tokenizer vocab size: ", tokenizer_midi.vocab.count)
    #text = preprocess_tokens(tokenizer_midi, midi_dir='session')
    with open("encoded.txt",'r',encoding="utf-8") as f:
        text = f.read()
        
    average_text_length = 0
    tokens_text = text.split("[SEP]") 
    for i in range(len(tokens_text)):
        average_text_length+=len(tokens_text[i])
    average_text_length/=len(tokens_text)
    print("Average Token Length Initially: ", round(average_text_length,4))
    print()
    
    with open("encoded.txt",'w+',encoding="utf-8") as f:
        f.write(text)
    
    tokenizer_nlp = Tokenizer.from_file("data/vocab_bpe.json")
    
    tokenizer_nlp = train_tokenizer(vocab_size=300, alg='BPE')
    
    #tokenizer_nlp.save("data/vocab_bpe_1000.json")
    
    with open("encoded.txt",'r',encoding="utf-8") as f:
        x = f.read()
    
    x1 = x.split("[SEP]")
    encoded_tokens_list = []
    average_tokens_length = 0
    
    for i in range(len(x1[:1000])):
        encoded_tokens_list.append(tokenizer_nlp.encode(x1[i]).ids)
        average_tokens_length+=len(encoded_tokens_list[-1])
        
    np.save("./data/train_bpe.npy", np.array(encoded_tokens_list))
    average_tokens_length/=len(encoded_tokens_list)
    print("Average Token Length BPE: ", round(average_tokens_length,4))
    print()
    
    #tokenizer_nlp = train_tokenizer(vocab_size=300, alg='UNI')
    
    tokenizer_nlp = Tokenizer.from_file("data/vocab_uni.json")
    #tokenizer_nlp.save("data/vocab_uni_1000.json")
    
    with open("encoded.txt",'r',encoding="utf-8") as f:
        x = f.read()
    
    x1 = x.split("[SEP]")
    encoded_tokens_list = []
    average_tokens_length = 0
    
    for i in range(len(x1[:1000])):
        encoded_tokens_list.append(tokenizer_nlp.encode(x1[i]).ids)
        average_tokens_length+=len(encoded_tokens_list[-1])

    np.save("./data/train_uni.npy", np.array(encoded_tokens_list))
    average_tokens_length/=len(encoded_tokens_list)
    print("Average Token Length Unicode: ", round(average_tokens_length,4))
    print()
    
    """
    tokenizer_nlp = train_tokenizer(vocab_size=300, alg='WPC')
    
    tokenizer_nlp.save("data/vocab_wpc.json")
    
    with open("encoded.txt",'r',encoding="utf-8") as f:
        x = f.read()
    
    x1 = x.split("[SEP]")
    encoded_tokens_list = []
    average_tokens_length = 0
    
    for i in tqdm(range(len(x1[100]))):
        encoded_tokens_list.append(tokenizer_nlp.encode(x1[i]).ids)
        average_tokens_length+=len(encoded_tokens_list[-1])
    average_tokens_length/=len(encoded_tokens_list)
    print("Average Token Length WordPiece: ", round(average_tokens_length,4))
    print()
    """


