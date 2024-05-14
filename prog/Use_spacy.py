import re
import glob
import spacy
import json
import os
import csv
import shutil
import warnings
warnings.simplefilter("ignore")
#TODO: gérer warnings
# from generic_tools import *
from optparse import OptionParser

def get_parser():
    """Returns a command line parser
    Returns:
        OptionParser. The command line parser
    """
    parser = OptionParser()
    parser.add_option("-d", "--data_path", dest="data_path",
                      help="""Chemin vers les fichiers txt (exemple DATA/*)""", type="string", default="../DATA/")
    parser.add_option('-F', '--Force', help='Recalculer les distances même si déjà faites', action='store_true', default = False)
    return parser

parser = get_parser()
options, _ = parser.parse_args()
path_corpora = options.data_path
print("")
print("-"*40)
print(f"Path corpora : '{path_corpora}'")
print("--> pour spécifier un autre chemin utiliser l'option -d")
print("-"*40)

def lire_fichier (chemin, is_json = False):
    f = open(chemin , encoding = 'utf−8')
    if is_json==False:
      chaine = f.read()
    else:
      chaine = json.load(f)
    f.close ()
    return chaine

def stocker( chemin, contenu, is_json=False, verbose =False):
    if verbose==True:
      print(f"  Output written in {chemin}")
    w = open(chemin, "w")
    if is_json==False:
      w.write(contenu)
    else:
      w.write(json.dumps(contenu , indent = 2, ensure_ascii=False))
    w.close()


def dico_resultats(texte, nlp =""):
    nlp.max_length = 1500000  # or any large value, as long as you don't run out of RAM
    #nlp=spacy.load("fr_core_news_sm")):
    if nlp == "":
      try:
        nlp = spacy.load("fr_core_news_sm")
      except:
        cmd = "python3 -m spacy download fr_core_news_sm"
        os.system(cmd)
        nlp = spacy.load("fr_core_news_sm")
        
    doc = nlp(texte)
    dico_resultats ={}
    i=0
    for ent in doc.ents:
        entite="entite_"+str(i)
        dico_resultats[entite]={}
        dico_resultats[entite]["label"] =ent.label_
        dico_resultats[entite]["text"] = ent.text
        dico_resultats[entite]["jalons"] = [ent.start_char, ent.end_char]
        i=i+1
    return (dico_resultats)


def bio_spacy(texte, nlp=""):
    nlp.max_length = 1500000  # or any large value, as long as you don't run out of RAM
    # nlp=spacy.load("fr_core_news_sm")):
    if nlp == "":
        try:
            nlp = spacy.load("fr_core_news_sm")
        except:
            cmd = "python3 -m spacy download fr_core_news_sm"
            os.system(cmd)
            nlp = spacy.load("fr_core_news_sm")

    doc = nlp(texte)
    liste_bio = []
    i = 0
    for ent in doc.ents:
        liste_bio.append([doc[i].text, doc[i].ent_iob_, doc[i].ent_type_])
        i = i + 1
    return (liste_bio)



for modele in ["sm"]:
# for modele in ["sm", "md", "lg"]:
    liste_subcorpus = glob.glob("%s/*/"%path_corpora)
    if len(liste_subcorpus)==0:
      print(f"Pas de dossier trouvé dans {path_corpora}, traitement terminé")
      exit()
    print("Starting with modèle %s"%modele)
    nom_complet_modele = "fr_core_news_%s"%modele
    try:
        nlp = spacy.load(nom_complet_modele)
    except:
        cmd = f"python3 -m spacy download {nom_complet_modele}"
        os.system(cmd)
        nlp = spacy.load(nom_complet_modele)
    nom_modele = f"spacy-{modele}"

    for subcorpus in liste_subcorpus:
        print("  Processing %s"%subcorpus)
        liste_txt = glob.glob("%s/*_REF/*.txt"%subcorpus)
        liste_txt+=  glob.glob("%s/OCR/*/*.txt"%subcorpus)
        print("  nombre de fichiers txt trouvés :",len(liste_txt))
        for path in liste_txt: 
            dossiers  = re.split("/", path)[:-1]
            nom_txt = re.split("/", path)[-1]
            path_ner = "/".join(dossiers)+"/NER"
            os.makedirs(path_ner, exist_ok = True)
            #Ajouter la version de spacy le chemin ressemble : auteur_titre-titre_ocr_spaCymodel-versionspaCy.json
            path_output = "%s/%s_%s.json"%(path_ner, nom_txt, nom_modele)
            # Pour le format bio
            path_output_bio = "%s/%s_%s.bio" % (path_ner, nom_txt, nom_modele)

            print(path_output)
            if os.path.exists(path_output)==True:
              if options.Force ==True:
                print("  Recomputing :",path_output)
              else:
                print("Already DONE : ", path_output)
                continue
            texte = lire_fichier(path)
            entites = dico_resultats(texte, nlp)
            stocker(path_output, entites, is_json=True)

            # Pour le format bio
            entites_bio=bio_spacy(texte, nlp)
            with open(path_output_bio, 'w', newline='') as file:
                writer = csv.writer(file, delimiter=';', quotechar='|')
                writer.writerows(entites_bio)
                # writer.writerows([["Alice", 23], ["Bob", 27]])
        ### Penser à comment lancer compute_distances
