import random
import spacy

nlp = spacy.load("en_core_web_sm")

def treeMirrorPre(in_str, p = 0):
    '''
    If p is set to 0. The sentence is unchanged.
    If p is set to 1. All sub trees are mirrored.
    else only a fraction of the subtrees are mirrored.
    '''
    class node:
        def __init__(self,val):
            self.val = val
            self.right = []
            self.left = []

        def addRight(self,vals):
            self.right = vals

        def addLeft(self,vals):
            self.left = vals

    toks = nlp(in_str)
    out_sent = ''
    for sent in toks.sents:
      root = None

      node_dict = {}

      for t in sent:
        node_dict[t.i] = node(t)

      count = {}
      last = None
      for t in sent:
        if str(t) not in ['.','!','?']:
          n_ = node_dict[t.i]
          n_right = [node_dict[_.i] for _ in t.rights]
          n_left = [node_dict[_.i] for _ in t.lefts]

          if random.random() > p:
            n_.addLeft(n_right)
            n_.addRight(n_left)
          else:
            n_.addLeft(n_left)
            n_.addRight(n_right)

          if t.dep_ == 'ROOT':
            root = n_
        else:
          last = str(t)


      def preorder(node):
          if node == None:
              return
          yield str(node.val)

          for t in node.left:
            for x in preorder(t):
              yield x

          for t in node.right:
            for x in preorder(t):
              yield x

      out_sent += ' ' + ' '.join(_ for _ in preorder(root)).replace(last, '') + ' ' +last
    out_sent = out_sent.replace('  ',' ')
    return out_sent

def treeMirrorPo(in_str, p = 0):
    '''
    If p is set to 0. The sentence is unchanged.
    If p is set to 1. All sub trees are mirrored.
    else only a fraction of the subtrees are mirrored.
    '''
    class node:
        def __init__(self,val):
            self.val = val
            self.right = []
            self.left = []

        def addRight(self,vals):
            self.right = vals

        def addLeft(self,vals):
            self.left = vals

    toks = nlp(in_str)
    out_sent = ''
    for sent in toks.sents:
      root = None

      node_dict = {}

      for t in sent:
        node_dict[t.i] = node(t)

      count = {}
      last = None
      for t in sent:
        if str(t) not in ['.','!','?']:
          n_ = node_dict[t.i]
          n_right = [node_dict[_.i] for _ in t.rights]
          n_left = [node_dict[_.i] for _ in t.lefts]

          if random.random() > p:
            n_.addLeft(n_right)
            n_.addRight(n_left)
          else:
            n_.addLeft(n_left)
            n_.addRight(n_right)

          if t.dep_ == 'ROOT':
            root = n_
        else:
          last = str(t)

      def postorder(node):
          if node == None:
              return

          for t in node.left:
            for x in postorder(t):
              yield x

          for t in node.right:
            for x in postorder(t):
              yield x
          yield str(node.val)
      out_sent += ' ' + ' '.join(_ for _ in postorder(root)).replace(last, '') + ' ' +last
    out_sent = out_sent.replace('  ',' ')
    return out_sent

def treeMirrorIn(in_str, p = 0):
    '''
    If p is set to 0. The sentence is unchanged.
    If p is set to 1. All sub trees are mirrored.
    else only a fraction of the subtrees are mirrored.
    '''
    class node:
        def __init__(self,val):
            self.val = val
            self.right = []
            self.left = []

        def addRight(self,vals):
            self.right = vals

        def addLeft(self,vals):
            self.left = vals

    toks = nlp(in_str)
    out_sent = ''
    for sent in toks.sents:
      root = None

      node_dict = {}

      for t in sent:
        node_dict[t.i] = node(t)

      count = {}
      last = None
      for t in sent:
        if str(t) not in ['.','!','?']:
          n_ = node_dict[t.i]
          n_right = [node_dict[_.i] for _ in t.rights]
          n_left = [node_dict[_.i] for _ in t.lefts]

          if random.random() > p:
            n_.addLeft(n_right)
            n_.addRight(n_left)
          else:
            n_.addLeft(n_left)
            n_.addRight(n_right)

          if t.dep_ == 'ROOT':
            root = n_
        else:
          last = str(t)


      def inorder(node):
          if node == None:
              return

          for t in node.left:
            for x in inorder(t):
              yield x

          yield str(node.val)
          for t in node.right:
            for x in inorder(t):
              yield x
      out_sent += ' ' + ' '.join(_ for _ in inorder(root)).replace(last, '') + ' ' +last
    out_sent = out_sent.replace('  ',' ')
    return out_sent

def rotateAroundRoot(in_str):
    toks = nlp(in_str)
    root_ind = 0
    final = []
    for sent in toks.sents:
      for t in sent:
          if t.dep_ == 'ROOT':
              root_ind = t.i
              break
      temp_str = [str(_) for _ in toks]
      out_str = temp_str[root_ind+1:-1] + [temp_str[root_ind]] + temp_str[:root_ind] + [temp_str[-1]]
      final.append(' '.join(out_str))
    return ' '.join(final)

def injectNonce(in_str):
    pass #to do

def reversed(in_str):
  toks = nlp(in_str)
  out_str = [str(toks[_]) for _ in range(len(toks)-2,-1,-1)] + [str(toks[-1])]
  return ' '.join(out_str)

def verb2andLast(in_str):
    pass #to do

def functionalShuffle(in_str, func_ = ['DET','ADP']):
  toks = nlp(in_str)
  func = []
  out_str = []
  for t in toks:
    if t.pos_ in func_:
      func.append(str(t))

  random.shuffle(func)
  for t in toks:
    if t.pos_ in ['DET', 'ADP']:
      out_str.append(func.pop(0))
    else:
      out_str.append(str(t))
  return ' '.join(out_str)

def nounVerbMismatched(in_str):
    toks = nlp(in_str)
    verb_stack = []
    noun_stack = []
    out_str = ['' for _ in range(len(toks))]
    untouch = ['' for _ in range(len(toks))]
    for t in toks:
        if t.pos_ == 'VERB':
            verb_stack.append(t)
    for np in toks.noun_chunks:
        noun_stack.append(np)

    while len(verb_stack) and len(noun_stack):
        last_noun = noun_stack.pop(0)
        last_verb = verb_stack.pop()
        n_split = str(last_noun).split()

        out_str[last_verb.i] = str(last_noun)
        out_str[last_noun.start] = str(last_verb)

        untouch[last_noun.start]  = str(last_noun)
        untouch[last_verb.i] = str(last_verb)

    if len(verb_stack):
      for t in verb_stack:
        out_str[t.i] = str(t)
        untouch[t.i] = str(t)
    if len(noun_stack):
      for t in noun_stack:
        out_str[t.start] = str(t)
        untouch[t.start] = str(t)
    i=0
    while i <len(toks):
      if untouch[i] == '':
        out_str[i] = str(toks[i])
        i+=1
      else:
        i+=len(untouch[i].split())
    temp = ' '.join(out_str)
    out_str = temp.replace('  ', ' ')
    return out_str

def adverbVerbSwap(in_str):
    toks = nlp(in_str)
    verb_stack = []
    adverb_stack = []
    out_str = ['' for _ in range(len(toks))]
    for t in toks:
        if t.pos_ == 'VERB':
          verb_stack.append(t)
        if t.pos_ == 'ADV':
          adverb_stack.append(t)

    while len(verb_stack) and len(adverb_stack):
        last_adverb = adverb_stack.pop()
        last_verb = verb_stack.pop()
        n_split = str(last_adverb).split()

        out_str[last_verb.i] = str(last_adverb)
        out_str[last_adverb.i] = str(last_verb)

    if len(verb_stack):
      for t in verb_stack:
        out_str[t.i] = str(t)
    if len(adverb_stack):
      for t in adverb_stack:
        out_str[t.i] = str(t)
    i=0
    while i <len(toks):
      if out_str[i] == '':
        out_str[i] = str(toks[i])
      i+=1
    return ' '.join(out_str)

def nounAdjSwap(in_str):
    toks = nlp(in_str)
    noun_stack = []
    adj_stack = []
    out_str = ['' for _ in range(len(toks))]
    untouch = ['' for _ in range(len(toks))]
    for t in toks:
        if t.pos_ == 'ADJ':
            adj_stack.append(t)
        if t.pos_ == 'NOUN':
          noun_stack.append(t)

    while len(adj_stack) and len(noun_stack):
        last_noun = noun_stack.pop()
        last_adj = adj_stack.pop(0)
        n_split = str(last_noun).split()

        out_str[last_adj.i] = str(last_noun)
        out_str[last_noun.i] = str(last_adj)

        untouch[last_noun.i]  = str(last_noun)
        untouch[last_adj.i] = str(last_adj)
    i=0
    while i <len(toks):
      if untouch[i] == '':
        out_str[i] = str(toks[i])
        i+=1
      else:
        i+=len(untouch[i].split())
    temp = ' '.join(out_str)
    out_str = temp.replace('  ', ' ')
    return out_str

def nounVerbSwap(in_str):
    toks = nlp(in_str)
    verb_stack = []
    noun_stack = []
    out_str = ['' for _ in range(len(toks))]
    untouch = ['' for _ in range(len(toks))]
    for t in toks:
        if t.pos_ in ['VERB','AUX']:
            verb_stack.append(t)
    for np in toks.noun_chunks:
        noun_stack.append(np)

    while len(verb_stack) and len(noun_stack):
        last_noun = noun_stack.pop(0)
        last_verb = verb_stack.pop(0)
        n_split = str(last_noun).split()

        out_str[last_verb.i] = str(last_noun)
        out_str[last_noun.start] = str(last_verb)

        untouch[last_noun.start]  = str(last_noun)
        untouch[last_verb.i] = str(last_verb)

    if len(verb_stack):
      for t in verb_stack:
        out_str[t.i] = str(t)
        untouch[t.i] = str(t)
    if len(noun_stack):
      for t in noun_stack:
        out_str[t.start] = str(t)
        untouch[t.start] = str(t)
    i=0
    while i <len(toks):
      if untouch[i] == '':
        out_str[i] = str(toks[i])
        i+=1
      else:
        i+=len(untouch[i].split())
    temp = ' '.join(out_str)
    out_str = temp.replace('  ', ' ')
    return out_str

def verbSwaps(in_str):
    toks = nlp(in_str)
    verbs = []
    out_str = []
    for t in toks:
        if t.pos_ in ['VERB','AUX']:
            verbs+=[str(t)]
    random.shuffle(verbs)
    verbs_gen = (_ for _ in verbs)

    for t in toks:
        if t.pos_ in ['VERB','AUX']:
            out_str += [next(verbs_gen)]
        else:
            out_str+= [str(t)]

    return ' '.join(out_str)

def wordShuffle(in_str):
    str_l = [str(_) for _ in nlp(in_str)]
    random.shuffle(str_l[:-1])
    return ' '.join(str_l) + ' ' + str(str_l[-1])

def shuffleHalvesFirst(in_str):
    str_l = [str(_) for _ in nlp(in_str)]

    out_str = wordShuffle(' '.join(str_l[:len(str_l)//2])) + ' '.join(str_l[len(str_l)//2:])

    return out_str

def shuffleHalvesLast(in_str):
    str_l = [str(_) for _ in nlp(in_str)]

    out_str = ' '.join(str_l[:len(str_l)//2]) + wordShuffle(' '.join(str_l[len(str_l)//2:- 1])) + ' ' +str(str_l[-1])

    return out_str

def verbAtBeginning(in_str):
  toks = nlp(in_str)
  out_str = []
  flag = True
  for t in toks:
      if flag:
          if t.pos_ == 'VERB':
              out_str = [str(t)] + out_str
              flag == False
              continue
      out_str += [str(t)]
  return ' '.join(out_str)

def nounSwaps(in_str):
  toks = nlp(in_str)
  noun_stack = []
  out_str = ['' for _ in range(len(toks))]
  untouch = ['' for _ in range(len(toks))]
  for np in toks.noun_chunks:
      noun_stack.append(np)

  unshuffled = [_ for _ in noun_stack]
  random.shuffle(noun_stack)

  while len(noun_stack):
    noun_mis = noun_stack.pop(0)
    noun_real = unshuffled.pop(0)

    out_str[noun_real.start] = str(noun_mis)

    untouch[noun_real.start]  = str(noun_real)
  i=0
  while i <len(toks):
    if untouch[i] == '':
      out_str[i] = str(toks[i])
      i+=1
    else:
      i+=len(untouch[i].split())

  temp = ' '.join(out_str)
  out_str = temp.replace('  ', ' ')
  return out_str

def conjunctionShuffle(in_str):
  toks = nlp(in_str)
  conjs = []
  out_str = []
  for t in toks:
      if 'CONJ' in str(t.pos_):
          conjs+=[str(t)]
  random.shuffle(conjs)
  conjs_gen = (_ for _ in conjs)

  for t in toks:
      if 'CONJ' in t.pos_:
          out_str += [next(conjs_gen)]
      else:
          out_str+= [str(t)]

  return ' '.join(out_str)
