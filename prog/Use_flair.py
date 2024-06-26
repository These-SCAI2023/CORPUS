from optparse import OptionParser
import re
import glob
from pathlib import Path
import json
import os
import csv
from flair.data import Sentence
from flair.models import SequenceTagger
import torch


def get_parser():
    """Returns a command line parser
    Returns:
        OptionParser. The command line parser
    """
    parser = OptionParser()
    parser.add_option("-d", "--data_path", dest="data_path",
                      help="""Chemin vers les fichiers txt (exemple DATA/*)""", type="string", default="../DATA/")
    parser.add_option('-F', '--Force', help='Recalculer les distances même si déjà faites',
                      action='store_true', default=False)
    return parser


parser = get_parser()
options, _ = parser.parse_args()
path_corpora = options.data_path
print("")
print("-"*40)
print(f"Path corpora : '{path_corpora}'")
print("--> pour spécifier un autre chemin utiliser l'option -d")
print("-"*40)


def lire_fichier(chemin, is_json=False):
    with open(chemin, encoding='utf-8') as f:
        if not is_json:
            return f.read()
        else:
            return json.load(f)


def stocker(chemin, contenu, is_json=False, verbose=False):
    with open(chemin, "w", encoding='utf-8') as w:
        if is_json:
            w.write(json.dumps(contenu, indent=2, ensure_ascii=False))
        else:
            w.write(contenu)
    if verbose:
        print(f"  Output written in {chemin}")


def chunk_text(text, chunk_size):
    """Splits text into chunks of specified size."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size
    return chunks


def get_entity(entity_dict: dict) -> dict:
    """reformat flair entity dict.

    Args:
        entity_dict (dict): flair entity dict like {"text": "", "start_pos": "", ... "labels": [{"value":...}] or []}
    """
    label: str = "O" if entity_dict["labels"] == [
    ] else entity_dict["labels"][0]["value"]

    return {
        'text': entity_dict["text"],
        'start_pos': entity_dict["start_pos"],
        'end_pos': entity_dict["end_pos"],
        'label': label
    }


def flair_ner(text, tagger, chunk_size=512):
    chunks = chunk_text(text, chunk_size)
    entities = []
    for chunk in chunks:
        sentence = Sentence(chunk)
        tagger.predict(sentence)
        # print(sentence.to_dict(tag_type='ner'))
        for entity in sentence.to_dict(tag_type='ner')["entities"]:
            print(entity)
            entities.append(get_entity(entity_dict=entity))
    return entities


if __name__ == "__main__":
    tagger = SequenceTagger.load('ner')
    tagger.to('cpu')  # Ensure model runs on CPU

    liste_subcorpus = list(Path(path_corpora).glob("*"))
    print(liste_subcorpus)
    print(os.getcwd())
    if len(liste_subcorpus) == 0:
        print(f"Pas de dossier trouvé dans {path_corpora}, traitement terminé")
        exit()

    for subcorpus in liste_subcorpus:
        print("  Processing %s" % subcorpus)
        liste_txt = glob.glob("%s/*_REF/*.txt" % subcorpus)
        liste_txt += glob.glob("%s/OCR/*/*.txt" % subcorpus)
        print("  nombre de fichiers txt trouvés :", len(liste_txt))
        for path in liste_txt:
            dossiers = re.split("/", path)[:-1]
            nom_txt = re.split("/", path)[-1]
            path_ner = os.path.join(*dossiers, "NER")
            os.makedirs(path_ner, exist_ok=True)
            path_output = f"{path_ner}/{nom_txt}_flair.json"
            if os.path.exists(path_output) and not options.Force:
                print("Already DONE : ", path_output)
                continue

            texte = lire_fichier(path)
            entites = flair_ner(texte, tagger)
            stocker(path_output, entites, is_json=True)
