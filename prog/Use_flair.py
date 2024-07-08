from optparse import OptionParser
import re
import glob
from pathlib import Path
import json
import os
import csv
from flair.data import Sentence
from flair.models import SequenceTagger
import flair
from itertools import chain
# import torch


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


def stocker(chemin: str, contenu, is_json=False, verbose=False, delimiter: str = " ", quotechar: str = "|"):
    if is_json:
        with open(chemin, "w", encoding='utf-8') as w:
            w.write(json.dumps(contenu, indent=2, ensure_ascii=False))
    else:
        with open(chemin, mode='w', newline='', encoding='utf-8') as w:
            writer = csv.writer(w, delimiter=delimiter, quotechar=quotechar)
            writer.writerows(contenu)
    if verbose:
        print(f"  Output written in {chemin}")


def chunk_text(text, chunk_size) -> list[list]:
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
    print(entity_dict)
    return {
        'label': label,
        'text': entity_dict["text"],
        'jalons': [entity_dict["start_pos"], entity_dict["end_pos"]]
    }


def generate_bio_tags(entity: dict, sep: str = " ") -> list[list[str]]:
    """generate bio tags (B/I) for potentially multiword entities.
    """
    words: list[str] = entity["text"].split(sep)
    print(words)
    bio_entity_list: list[list] = [
        [words.pop(0), f"B-{entity['labels'][0]['value']}"]]
    bio_entity_list.extend(
        [[word, f"I-{entity['labels'][0]['value']}"] for word in words])
    print(bio_entity_list)
    return bio_entity_list


def dico_resultats(text, tagger: SequenceTagger, chunk_size: int = 512):
    chunks: list[list] = chunk_text(text=text, chunk_size=chunk_size)
    all_ner_results: list[dict] = []
    for chunk in chunks:
        sentence = Sentence(chunk)
        tagger.predict(sentence)
        all_ner_results.extend([get_entity(entity_dict=entity)
                                for entity in sentence.to_dict(tag_type='ner')["entities"]])
        # print(all_ner_results)
    return {f"entite_{i}": ent for i, ent in enumerate(all_ner_results)}


def bio_flair(text, tagger: SequenceTagger, chunk_size: int = 512):
    chunks: list[list] = chunk_text(text=text, chunk_size=chunk_size)
    all_ner_results: list[dict] = []
    for chunk in chunks:
        sentence = Sentence(chunk)
        tagger.predict(sentence)
        all_ner_results.extend(chain.from_iterable([generate_bio_tags(entity=entity)
                               for entity in sentence.to_dict(tag_type='ner')["entities"]]))
        print(all_ner_results)
    return all_ner_results


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
        print(f"  Processing {subcorpus}")
        liste_txt = glob.glob(f"{subcorpus}/*_REF/*.txt")
        liste_txt += glob.glob(f"{subcorpus}/OCR/*/*.txt")
        print("  nombre de fichiers txt trouvés :", len(liste_txt))
        for path in liste_txt:
            dossiers = re.split("/", path)[:-1]
            nom_txt = re.split("/", path)[-1]
            path_ner = os.path.join(*dossiers, "NER")
            os.makedirs(path_ner, exist_ok=True)
            path_output: str = f"{path_ner}/{nom_txt}_flair-{flair.__version__}.json"
            path_output_bio: str = f"{path_ner}/{nom_txt}_flair-{flair.__version__}.bio"

            if os.path.exists(path_output) and not options.Force:
                print("Already DONE : ", path_output)
                continue

            texte = lire_fichier(path)
            # entites = flair_ner(texte, tagger)
            entites = dico_resultats(texte, tagger)
            print(entites)
            stocker(path_output, entites, is_json=True)
            bio_entites = bio_flair(text=texte, tagger=tagger)
            print(bio_entites)
            stocker(chemin=path_output_bio, contenu=bio_entites, is_json=False)
