from nltk.translate.bleu_score import sentence_bleu as bleu_score
#from bleurt import score
#bleurt_ops = score.create_bleurt_ops('/home/pparth2/scratch/UMT/UMT/Results/cached/bleurt-base-128')
from bert_score import score

def bleurt_score(a,b, weights=None, lang=None):
    out = bleurt_ops(references=a, candidates=b)
    out_f = [float(_) for _ in out["predictions"]]
    return out_f

def bertscore(a,b, weights=None, lang=None):
    _,_,f1 = score(a, b, lang=lang, verbose=False)
    out_f = [float(_) for _ in f1]
    return out_f

def bleu(a,b, weights = (1.0,), lang=None):
    a = [a]
    return bleu_score(a, b, weights)

def levenshtein(a,b, weights = None, lang=None):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]
