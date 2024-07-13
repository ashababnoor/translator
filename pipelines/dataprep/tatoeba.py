import pandas as pd 
import os  


# Setting the directories
DATAPREP_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINES_DIR = os.path.dirname(DATAPREP_DIR)
ROOT_DIR = os.path.dirname(PIPELINES_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TATOEBA_DATA_DIR = os.path.join(DATA_DIR, 'tatoeba')


# Setting the file names
EN_FILE_NAME = 'eng_sentences.tsv'
BN_FILE_NAME = 'ben_sentences.tsv'
TRANSLATIONS_FILE_NAME = 'sentences_base.csv'
BN_TO_EN_TRANSLATIONS_FILE_NAME = 'bn_to_en_translations.csv'

EN_FILE_PATH = os.path.join(TATOEBA_DATA_DIR, EN_FILE_NAME)
BN_FILE_PATH = os.path.join(TATOEBA_DATA_DIR, BN_FILE_NAME)
TRANSLATIONS_FILE_PATH = os.path.join(TATOEBA_DATA_DIR, TRANSLATIONS_FILE_NAME)
BN_TO_EN_TRANSLATIONS_FILE_PATH = os.path.join(TATOEBA_DATA_DIR, BN_TO_EN_TRANSLATIONS_FILE_NAME)


# Reading the files
en_sentences = pd.read_csv(EN_FILE_PATH, sep='\t', header=None)
bn_sentences = pd.read_csv(BN_FILE_PATH, sep='\t', header=None)
translations = pd.read_csv(TRANSLATIONS_FILE_PATH, sep='\t', header=None)


# Setting the column names
sentence_file_columns = ['sentence_id', 'lang', 'text']
translation_file_columns = ['sentence_id_1', 'sentence_id_2']

en_sentences.columns = sentence_file_columns
bn_sentences.columns = sentence_file_columns
translations.columns = translation_file_columns


# Filtering the English and Bengali sentences
en_sentences = en_sentences[en_sentences['lang'] == 'eng']
bn_sentences = bn_sentences[bn_sentences['lang'] == 'ben']


# Getting the sentence ids and filtering the translations
en_sentence_ids = set(en_sentences['sentence_id'])
bn_sentence_ids = set(bn_sentences['sentence_id'])

translations_en_to_bn = translations[translations['sentence_id_1'].isin(en_sentence_ids) & translations['sentence_id_2'].isin(bn_sentence_ids)]
translations_bn_to_en = translations[translations['sentence_id_1'].isin(bn_sentence_ids) & translations['sentence_id_2'].isin(en_sentence_ids)]


# Reversing the translations and concatenating them
translations_en_to_bn_reversed = translations_en_to_bn.copy()
translations_en_to_bn_reversed = translations_en_to_bn_reversed.rename(columns={'sentence_id_1': 'sentence_id_2', 'sentence_id_2': 'sentence_id_1'})
translations_en_to_bn_reversed = translations_en_to_bn_reversed[['sentence_id_1', 'sentence_id_2']]

merged_bn_to_en = pd.concat([translations_bn_to_en, translations_en_to_bn_reversed])
unique_bn_to_en_translations = merged_bn_to_en.drop_duplicates(subset=['sentence_id_1', 'sentence_id_2'])


# Merge the sentences with the translations

# Step 1: Merge to get Bengali sentences
bn_to_en_with_sentences = pd.merge(
    unique_bn_to_en_translations, 
    bn_sentences, 
    left_on='sentence_id_1', 
    right_on='sentence_id', 
    how='left'
)

# Step 2: Merge to get English sentences
bn_to_en_with_sentences = pd.merge(
    bn_to_en_with_sentences, 
    en_sentences, 
    left_on='sentence_id_2', 
    right_on='sentence_id', 
    how='left', 
    suffixes=('_bn', '_en')
)

# Step 3: Select relevant columns (assuming the actual sentences are in 'text' columns)
bn_to_en_with_sentences = bn_to_en_with_sentences[['sentence_id_1', 'text_bn', 'sentence_id_2', 'text_en']]


# Saving the merged data
bn_to_en_with_sentences.to_csv(BN_TO_EN_TRANSLATIONS_FILE_PATH, sep=',', index=False)


# Calculating the statistics
total_bn_sentences = len(bn_sentences)
total_en_sentences = len(en_sentences)

bn_to_en_translations = len(translations_bn_to_en)
en_to_bn_translations = len(translations_en_to_bn)

total_bn_to_en_translations = len(merged_bn_to_en)
total_unique_bn_to_en_translations = len(unique_bn_to_en_translations)


# Generating the report
report = f"""
Total Bengali Sentences: {total_bn_sentences:,}
Total English Sentences: {total_en_sentences:,}

Bengali to English Translations: {bn_to_en_translations:,}
English to Bengali Translations: {en_to_bn_translations:,}

Total Bengali to English Translations: {total_bn_to_en_translations:,}
Total Unique Bengali to English Translations: {total_unique_bn_to_en_translations:,}

The merged data has been saved to: {BN_TO_EN_TRANSLATIONS_FILE_PATH}
"""

print(report)
