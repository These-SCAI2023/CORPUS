from optparse import OptionParser
import re
import glob
from pathlib import Path
import spacy
from more_itertools import chunked
import json
import os
import csv
import shutil
import warnings
warnings.simplefilter("ignore")
# TODO: gérer warnings
# from generic_tools import *

spacy_models: dict = {"fr": ["fr_core_news_sm", "fr_core_news_md", "fr_core_news_lg"],
                      "pt": ["pt_core_news_sm", "pt_core_news_md", "pt_core_news_lg"], "en": ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]}


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
    f = open(chemin, encoding='utf−8')
    if is_json == False:
        chaine = f.read()
    else:
        chaine = json.load(f)
    f.close()
    return chaine


def stocker(chemin, contenu, is_json=False, verbose=False):
    if verbose == True:
        print(f"  Output written in {chemin}")
    w = open(chemin, "w")
    if is_json == False:
        w.write(contenu)
    else:
        w.write(json.dumps(contenu, indent=2, ensure_ascii=False))
    w.close()


def chunk_text(text: str, chunk_size: int = 1024) -> list[str]:
    """Splits text into chunks of specified size."""
    chunks = chunked(text.split(), n=chunk_size)
    return [" ".join(chunk) for chunk in chunks]


def dico_resultats(texte, spacy_model: str = ""):
    try:
        nlp = spacy.load(spacy_model)
    except:
        cmd = f"python -m spacy download {spacy_model}"
        os.system(cmd)
        nlp = spacy.load(spacy_model)
    nlp.max_length = 50000000  # or any large value, as long as you don't run out of RAM
    doc = nlp(texte)
    dico_resultats = {}
    for i, ent in enumerate(doc.ents):
        entite = f"entite_{i}"
        dico_resultats[entite] = {}
        dico_resultats[entite]["label"] = ent.label_
        dico_resultats[entite]["text"] = ent.text
        dico_resultats[entite]["jalons"] = [ent.start_char, ent.end_char]
    return dico_resultats, nlp


def bio_spacy(texte, nlp="") -> list[list]:
    nlp.max_length = 50000000  # or any large value, as long as you don't run out of RAM
    # nlp=spacy.load("fr_core_news_sm")):
    if nlp == "":
        try:
            nlp = spacy.load("fr_core_news_sm")
        except:
            cmd = "python3 -m spacy download fr_core_news_sm"
            os.system(cmd)
            nlp = spacy.load("fr_core_news_sm")
    # chunks: list[str] = chunk_text(text=texte)
    # print(chunks)
    # print(len(chunks))
    # exit()
    doc = nlp(texte)
    liste_bio: list = [[doc[i].text, f"{doc[i].ent_iob_}{'-' if  doc[i].ent_iob_ != 'O' else ''}{doc[i].ent_type_}"
                        ] for i, ent in enumerate(doc)]
    print(liste_bio)
    """
    for chunk in chunks:
        doc = nlp(chunk)
        print(f"doc: {doc}")
        for i, ent in enumerate(doc.ents):
            print(i, ent)
        liste_bio.append([[doc[i].text, doc[i].ent_iob_, doc[i].ent_type_]
                          for i, ent in enumerate(doc)])
        print(liste_bio)
        # exit()
    """
    return liste_bio


if __name__ == "__main__":
    do_json: bool = True
    nlp = "fr"
    # nlp = "pt"
    # nlp = "en"
    for spacy_model in spacy_models[nlp]:
        print(spacy_model)
        modele = spacy_model[-2:]
        liste_subcorpus = list(Path(path_corpora).glob("*"))
        print(liste_subcorpus)
        print(os.getcwd())
        if len(liste_subcorpus) == 0:
            print(
                f"Pas de dossier trouvé dans {path_corpora}, traitement terminé")
            exit()
        print(f"Starting with modèle {modele}")
        # nom_complet_modele = "fr_core_news_%s" % modele
        # try:
        #    nlp = spacy.load(nom_complet_modele)
        # except:
        #    cmd = f"{nom_complet_modele}"
        #    os.system(cmd)
        #    nlp = spacy.load(nom_complet_modele)
        nom_modele = f"spacy-{modele}"

        for subcorpus in liste_subcorpus:
            print(f"  Processing {subcorpus}")
            liste_txt = glob.glob(f"{subcorpus}/*_REF/*.txt")
            liste_txt += glob.glob(f"{subcorpus}/*OCR/*/*.txt")
            print("  nombre de fichiers txt trouvés :", len(liste_txt))
            for path in liste_txt:
                dossiers = re.split("/", path)[:-1]
                nom_txt = re.split("/", path)[-1]
                # path_ner = "/".join(dossiers)+"/NER"
                path_ner = os.path.join(*dossiers, "NER")
                os.makedirs(path_ner, exist_ok=True)
                # format json
                path_output = f"{path_ner}/{nom_txt}_{nom_modele}-{spacy.__version__}.json"
                print(path_output)
                # Pour le format bio
                path_output_bio = f"{path_ner}/{nom_txt}_{nom_modele}-{spacy.__version__}.bio"
                print(path_output_bio)
                # exit()
                if os.path.exists(path_output) == True:
                    if options.Force == True:
                        print("  Recomputing :", path_output)
                    else:
                        print("Already DONE : ", path_output)
                        continue
                texte = lire_fichier(path)
                if do_json:
                    entites, nlp_new = dico_resultats(
                        texte, spacy_model=spacy_model)
                    stocker(path_output, entites, is_json=True)

                # Pour le format bio
                entites_bio: list[list] = bio_spacy(texte, nlp_new)
                concat_bio_path: str = f"{path_ner}/{nom_txt}_{nom_modele}-{spacy.__version__}.bio"
                # for ent in entites_bio:
                #    print(ent)
                # path_output_bio = f"{path_ner}/{nom_txt}_{nom_modele}-{spacy.__version__}_chunk_{i}.bio"
                # with open(path_output_bio, 'a', #newline='') as file:
                # writer = csv.writer(file, delimiter=';', quotechar='|')
                # writer.writerows(ent)
                with open(concat_bio_path, "w",  newline='') as file:
                    writer = csv.writer(file, delimiter=' ', quotechar='|')
                    writer.writerows(entites_bio)

                    # writer.writerows([["Alice", 23], ["Bob", 27]])
            # Penser à comment lancer compute_distances
