# import spacy
# from spacy.matcher import Matcher
# from spacy.language import Language
# import pandas as pd
# from Simstring_Spell_Correction_English_Words import Simstring_Spell_Correction_Check_English_Words
# from spacy.util import filter_spans
#
# # Define custom NER labels and keywords
# CUSTOM_NER_PURCHASE_ORDER = "PURCHASE ORDER NUMBER"
# PURCHASE_ORDER_KEYWORDS = ["PO Number:", "PO #", "PO#", "P.O. Number", "PO Number", "PURCHASE ORDER NUMBER",
#                            "PURCHASEORDER NUMBER", "PURCHASE ORDERNUMER", "PURCHASE ORDER NO.", "PURCHASEORDER NO.",
#                            "PURCHASE ORDERNO.", "STANDARD PURCHASE ORDER", "PURCHASE ORDER #", "P.O. No.", "PO#",
#                            "PURCHASE ORDER NUMBER", "P.O.No.", "P.O."]
#
# CUSTOMER_NER_TRANS_NUMBER = "TRANS NUMBER"
# TRANS_NUMBER_KEYWORDS = ["Trans Number:", "Trans Number :", "TransNumber:", "TransNumber :", "Trans#", "Trans #"]
#
# CUSTOM_NER_LABEL_DATE = "INVOICE DATE"
# INVOICE_DATE_KEYWORDS = ["InvoiceDate", "Invoice Date", "Date Check Required", "DateCheckRequired",
#                          "DateCheck Required", "Date CheckRequired", "Date Requested", "DateRequested"]
#
# CUSTOM_NER_LABEL_NUMBER = "INVOICE NUMBER"
# INVOICE_NUMBER_KEYWORDS = ["InvoiceNumber", "Invoice No", "Invoice No.", "Invoice#","Invoice"
#                            "Invoice #", "INV#", "INV Number", "INV Number", "Invoice Number",
#                            "Invoice:", "Invoice :", "Invoice-", ["Invoice", "#", ":"], ["Invoice", "#", ":", " "],
#                            ['Invoice', 'Number', ':'], ['Invoice', 'Number', ':', " "], ["Invoice", ":"],
#                            ["Invoice", " ", ":"], ["Invoice", ":", " "], ["Invoice:"], ["Inv", "#"],
#                            ["Invoice", "ID", ":"], ["-", "Invoice", "-"], ["-", "Invoice"]]
#
# nlp = spacy.load("en_core_web_trf")
# matcher = Matcher(nlp.vocab)
#
#
# def create_patterns(keywords):
#     patterns = []
#     for keyword in keywords:
#         # Check if the keyword is a string or a list
#         if isinstance(keyword, str):
#             tokens = keyword.split()
#         else:
#             tokens = keyword
#
#         # Create a pattern for each token
#         pattern = [{"LOWER": token.lower()} for token in tokens]
#         patterns.append(pattern)
#
#     return patterns
#
#
# # Add patterns to matcher
# invoice_date_patterns = create_patterns(INVOICE_DATE_KEYWORDS)
# matcher.add(CUSTOM_NER_LABEL_DATE, invoice_date_patterns)
#
# invoice_number_patterns = create_patterns(INVOICE_NUMBER_KEYWORDS)
# matcher.add(CUSTOM_NER_LABEL_NUMBER, invoice_number_patterns)
#
# purchase_order_pattern = create_patterns(PURCHASE_ORDER_KEYWORDS)
# matcher.add(CUSTOM_NER_PURCHASE_ORDER, purchase_order_pattern)
#
# trans_number_pattern = create_patterns(TRANS_NUMBER_KEYWORDS)
# matcher.add(CUSTOMER_NER_TRANS_NUMBER, trans_number_pattern)
#
# pattern_invoice_hashtag_combined = [{"LOWER": {"REGEX": "invoice#"}}]
# matcher.add(CUSTOM_NER_LABEL_NUMBER, [pattern_invoice_hashtag_combined])
#
# # Define pattern for "invoice" followed by "#"
# pattern_invoice_followed_by_hashtag = [{"LOWER": "invoice"}, {"TEXT": "#"}]
# matcher.add(CUSTOM_NER_LABEL_NUMBER, [pattern_invoice_followed_by_hashtag])
#
#
# def correct_spelling(input_text, similarity_score=0.6):
#     input_words = [word for word in input_text.split()]
#     simstring_output = Simstring_Spell_Correction_Check_English_Words(input_words, similarity_score)
#     correction_map = {}
#     for correction in simstring_output:
#         try:
#             original_word = correction[0]['Input_Word']
#             corrected_word = correction[0]['Get_word']
#             correction_map[original_word] = corrected_word
#         except (IndexError, KeyError):
#             pass
#     corrected_input_words = [correction_map.get(word, word) for word in input_words]
#     corrected_input_text = " ".join(corrected_input_words)
#     return corrected_input_text
#
#
# @Language.component("custom_ner_component")
# def custom_ner_component(doc):
#     matches = matcher(doc)
#     new_ents = []
#     existing_entities = [(ent.start, ent.end) for ent in doc.ents]
#     for match_id, start, end in matches:
#         span = doc[start:end]
#         if span:
#             overlap = any(start >= ent_start and end <= ent_end for ent_start, ent_end in existing_entities)
#             if overlap:
#                 conflicting_ents = [ent for ent in doc.ents if not (ent.start < end and ent.end > start)]
#                 doc.ents = [ent for ent in doc.ents if not (ent.start >= start and ent.end <= end)]
#                 existing_entities = [(ent.start, ent.end) for ent in doc.ents]
#             new_ent = spacy.tokens.Span(doc, start, end, label=nlp.vocab.strings[match_id])
#             new_ents.append(new_ent)
#             existing_entities.append((start, end))
#
#     # Assuming `new_ents` is your list of new entities
#     new_ents = filter_spans(new_ents)
#     doc.ents = new_ents
#     return doc
#
#
# nlp.add_pipe("custom_ner_component", after="ner")
#
#
# def NER_regex(object_df, batch_size=5):
#     listall = []
#     lines = object_df['LineNum'].unique()
#     line_batches = [lines[i:i + batch_size] for i in range(0, len(lines), batch_size)]
#     batch_lengths = []
#     batch_count = 0
#     correctedbatchtextpara = ""
#     for batch_lines in line_batches:
#         batch_count += 1
#         batch_text = []
#         for line_num in batch_lines:
#             objects_on_line = object_df.loc[object_df['LineNum'] == line_num, 'Object'].tolist()
#             batch_length = [len(obj) for obj in objects_on_line]
#             batch_lengths.append(batch_length)
#             batch_text.append(" ".join(objects_on_line))
#
#         batch_text = "\n".join(batch_text)
#         corrected_batch_text = correct_spelling(batch_text.strip())
#         correctedbatchtextpara += corrected_batch_text + "\n"
#         print("correctedbatchtextpara", correctedbatchtextpara)
#         print("correctedbatchtextpara1", correctedbatchtextpara.strip())
#         doc = nlp(correctedbatchtextpara.strip())
#         ner_entities = [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char} for ent in
#                         doc.ents]
#
#         result = {"Object": batch_text, "NER": ner_entities, 'Batch_lengths': batch_lengths, 'Batch': batch_count}
#         listall.append(result)
#         batch_lengths = []
#     df = pd.DataFrame(listall)
#     return df
import spacy
from spacy.matcher import Matcher
from spacy.language import Language
import pandas as pd
from Simstring_Spell_Correction_English_Words import Simstring_Spell_Correction_Check_English_Words
from spacy.util import filter_spans

# Define custom NER labels and keywords
CUSTOM_NER_PURCHASE_ORDER = "PURCHASE ORDER NUMBER"
PURCHASE_ORDER_KEYWORDS = ["PO Number:", "PO #", "PO#", "P.O. Number", "PO Number", "PURCHASE ORDER NUMBER",
                           "PURCHASEORDER NUMBER", "PURCHASE ORDERNUMER", "PURCHASE ORDER NO.", "PURCHASEORDER NO.",
                           "PURCHASE ORDERNO.", "STANDARD PURCHASE ORDER", "PURCHASE ORDER #", "P.O. No.", "PO#",
                           "PURCHASE ORDER NUMBER", "P.O.No.", "P.O."]

CUSTOMER_NER_TRANS_NUMBER = "TRANS NUMBER"
TRANS_NUMBER_KEYWORDS = ["Trans Number:", "Trans Number :", "TransNumber:", "TransNumber :", "Trans#", "Trans #"]

CUSTOM_NER_LABEL_DATE = "INVOICE DATE"
INVOICE_DATE_KEYWORDS = ["InvoiceDate", "Invoice Date", "Date Check Required", "DateCheckRequired",
                         "DateCheck Required", "Date CheckRequired", "Date Requested", "DateRequested"]

CUSTOM_NER_LABEL_NUMBER = "INVOICE NUMBER"
INVOICE_NUMBER_KEYWORDS = ["InvoiceNumber", "Invoice No", "Invoice No.", "Invoice#", "Invoice",
                           "Invoice #", "INV#", "INV Number", "INV Number", "Invoice Number",
                           "Invoice:", "Invoice :", "Invoice-", ["Invoice", "#", ":"], ["Invoice", "#", ":", " "],
                           ['Invoice', 'Number', ':'], ['Invoice', 'Number', ':', " "], ["Invoice", ":"],
                           ["Invoice", " ", ":"], ["Invoice", ":", " "], ["Invoice:"], ["Inv", "#"],
                           ["Invoice", "ID", ":"], ["-", "Invoice", "-"], ["-", "Invoice"]]

nlp = spacy.load("en_core_web_trf")
matcher = Matcher(nlp.vocab)


def create_patterns(keywords):
    patterns = []
    for keyword in keywords:
        # Check if the keyword is a string or a list
        if isinstance(keyword, str):
            tokens = keyword.split()
        else:
            tokens = keyword

        # Create a pattern for each token
        pattern = [{"LOWER": token.lower()} for token in tokens]
        patterns.append(pattern)

    return patterns


# Add patterns to matcher
invoice_date_patterns = create_patterns(INVOICE_DATE_KEYWORDS)
matcher.add(CUSTOM_NER_LABEL_DATE, invoice_date_patterns)

invoice_number_patterns = create_patterns(INVOICE_NUMBER_KEYWORDS)
matcher.add(CUSTOM_NER_LABEL_NUMBER, invoice_number_patterns)

purchase_order_pattern = create_patterns(PURCHASE_ORDER_KEYWORDS)
matcher.add(CUSTOM_NER_PURCHASE_ORDER, purchase_order_pattern)

trans_number_pattern = create_patterns(TRANS_NUMBER_KEYWORDS)
matcher.add(CUSTOMER_NER_TRANS_NUMBER, trans_number_pattern)

pattern_invoice_hashtag_combined = [{"LOWER": {"REGEX": "invoice#"}}]
matcher.add(CUSTOM_NER_LABEL_NUMBER, [pattern_invoice_hashtag_combined])

# Define pattern for "invoice" followed by "#"
pattern_invoice_followed_by_hashtag = [{"LOWER": "invoice"}, {"TEXT": "#"}]
matcher.add(CUSTOM_NER_LABEL_NUMBER, [pattern_invoice_followed_by_hashtag])


def correct_spelling(input_text, similarity_score=0.6):
    input_words = [word for word in input_text.split()]
    simstring_output = Simstring_Spell_Correction_Check_English_Words(input_words, similarity_score)
    correction_map = {}
    for correction in simstring_output:
        try:
            original_word = correction[0]['Input_Word']
            corrected_word = correction[0]['Get_word']
            correction_map[original_word] = corrected_word
        except (IndexError, KeyError):
            pass
    corrected_input_words = [correction_map.get(word, word) for word in input_words]
    corrected_input_text = " ".join(corrected_input_words)
    return corrected_input_text


@Language.component("custom_ner_component")
def custom_ner_component(doc):
    matches = matcher(doc)
    new_ents = []
    existing_entities = [(ent.start, ent.end) for ent in doc.ents]
    for match_id, start, end in matches:
        span = doc[start:end]
        if span:
            overlap = any(start >= ent_start and end <= ent_end for ent_start, ent_end in existing_entities)
            if overlap:
                conflicting_ents = [ent for ent in doc.ents if not (ent.start < end and ent.end > start)]
                doc.ents = [ent for ent in doc.ents if not (ent.start >= start and ent.end <= end)]
                existing_entities = [(ent.start, ent.end) for ent in doc.ents]
            new_ent = spacy.tokens.Span(doc, start, end, label=nlp.vocab.strings[match_id])
            new_ents.append(new_ent)
            existing_entities.append((start, end))

    # Assuming `new_ents` is your list of new entities
    new_ents = filter_spans(new_ents)
    doc.ents = new_ents
    return doc


nlp.add_pipe("custom_ner_component", after="ner")


def NER_regex(object_df, batch_size=5):
    listall = []
    lines = object_df['LineNum'].unique()
    line_batches = [lines[i:i + batch_size] for i in range(0, len(lines), batch_size)]
    batch_count = 0
    for batch_lines in line_batches:
        batch_count += 1
        batch_text = []
        batch_lengths = []  # Reset batch lengths for each batch
        for line_num in batch_lines:
            objects_on_line = object_df.loc[object_df['LineNum'] == line_num, 'Object'].tolist()
            batch_length = [len(obj) for obj in objects_on_line]
            batch_lengths.append(batch_length)
            batch_text.append(" ".join(objects_on_line))

        batch_text = "\n".join(batch_text)
        corrected_batch_text = correct_spelling(batch_text.strip())
        doc = nlp(corrected_batch_text)
        ner_entities = [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char} for ent in
                        doc.ents]

        # Check if there are any NER entities detected
        if ner_entities:
            result = {"Object": batch_text, "NER": ner_entities, 'Batch_lengths': batch_lengths, 'Batch': batch_count}
        else:
            result = {"Object": batch_text, "NER": [], 'Batch_lengths': batch_lengths, 'Batch': batch_count}

        listall.append(result)

    df = pd.DataFrame(listall)
    return df

# Example usage:
# df = pd.DataFrame({'LineNum': [1, 1, 2, 2], 'Object': ['PO Number: 123', 'Date: 2023-05-01', 'Invoice Number: 456', 'Amount: $789']})
# result_df = NER_regex(df)
# print(result_df)
