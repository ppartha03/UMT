import random
import spacy

nlp = spacy.load("en_core_web_sm")

def treeMirrorPre(in_str, p = 0, nlp = None):
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

    s_l = [_ for _ in toks]
    assert len(s_l) > 5

    if toks[-1].pos_ != 'PUNCT':
      toks = nlp(in_str + ' .')
    out_sent = ''
    for sent in toks.sents:
      root = None

      node_dict = {}

      for t in sent:
        node_dict[t.i] = node(t)

      last = None
      for t in sent:
        if str(t) not in ['.',':','...','!','?',';']:
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
      assert root != None
      out_sent += ' ' + ' '.join(_ for _ in preorder(root)).replace(last, '') + ' ' +last
    out_sent = out_sent.replace('  ',' ')
    return out_sent

def treeMirrorPo(in_str, p = 0, nlp = None):
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
    s_l = [_ for _ in toks]
    assert len(s_l) > 5

    if toks[-1].pos_ != 'PUNCT':
      toks = nlp(in_str + ' .')
    out_sent = ''
    for sent in toks.sents:
      root = None

      node_dict = {}

      for t in sent:
        node_dict[t.i] = node(t)

      count = {}
      last = None
      for t in sent:
        if str(t) not in ['.','!','?',';']:
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

      assert root != None

      out_sent += ' ' + ' '.join(_ for _ in postorder(root)).replace(last, '') + ' ' +last
    out_sent = out_sent.replace('  ',' ')
    return out_sent

def treeMirrorIn(in_str, p = 0, nlp = None):
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

    s_l = [_ for _ in toks]
    assert len(s_l) > 5

    if toks[-1].pos_ != 'PUNCT':
      toks = nlp(in_str + ' .')

    out_sent = ''
    for sent in toks.sents:
      root = None

      node_dict = {}

      for t in sent:
        node_dict[t.i] = node(t)

      count = {}
      last = None
      for t in sent:
        if str(t) not in ['.','!','?',';']:
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
      assert root != None
      out_sent += ' ' + ' '.join(_ for _ in inorder(root)).replace(last, '') + ' ' +last
    out_sent = out_sent.replace('  ',' ')
    return out_sent

def rotateAroundRoot(in_str, nlp = None):
    toks = nlp(in_str)
    s_l = [_ for _ in toks]
    assert len(s_l) > 5

    if toks[-1].pos_ != 'PUNCT':
      toks = nlp(in_str + ' .')
    root_ind = None
    final = []
    len_sent = 0
    for sent in toks.sents:
      i = 0
      for t in sent:
          if t.dep_ == 'ROOT':
              root_ind = t.i
          i+=1
      root_ind -= len_sent
      len_sent += i
      temp_str = [str(_) for _ in sent]

      assert root_ind != None

      out_str = temp_str[root_ind+1:-1] + [temp_str[root_ind]] + temp_str[:root_ind] + [temp_str[-1]]
      final.append(' '.join(out_str))
    return ' '.join(final)

def injectNonce():
    check
    pass

def reversed(in_str, nlp = None):
  toks = nlp(in_str)
  s_l = [_ for _ in toks]
  assert len(s_l) > 5

  if toks[-1].pos_ != 'PUNCT':
    toks = nlp(in_str + ' .')
  out_str = [str(toks[_]) for _ in range(len(toks)-2,-1,-1)] + [str(toks[-1])]
  return ' '.join(out_str)

def functionalShuffle(in_str, func_ = ['DET','ADP'], nlp = None):
  toks = nlp(in_str)
  s_l = [_ for _ in toks]
  assert len(s_l) > 5

  if toks[-1].pos_ != 'PUNCT':
    toks = nlp(in_str + ' .')
  func = []
  out_str = []
  for t in toks:
    if t.pos_ in func_:
      func.append(str(t))

  assert len(func) > 1

  random.shuffle(func)
  for t in toks:
    if t.pos_ in ['DET', 'ADP']:
      out_str.append(func.pop(0))
    else:
      out_str.append(str(t))
  return ' '.join(out_str)

def nounVerbMismatched(in_str, nlp = None):
    toks = nlp(in_str)
    s_l = [_ for _ in toks]
    assert len(s_l) > 5

    if toks[-1].pos_ != 'PUNCT':
      toks = nlp(in_str + ' .')
    verb_stack = []
    noun_stack = []
    out_str = ['' for _ in range(len(toks))]
    untouch = ['' for _ in range(len(toks))]
    flag = True
    for t in toks:
        if t.pos_ == 'VERB':
            verb_stack.append(t)
    try:
        for np in toks.noun_chunks:
            noun_stack.append(np)
    except:
        flag = False
        for t in toks:
            if t.pos_ in ['NOUN','PRON']:
              noun_stack.append(t)

    assert len(noun_stack) >= 1 and len(verb_stack) >= 1

    while len(verb_stack) and len(noun_stack):
        last_noun = noun_stack.pop(0)
        last_verb = verb_stack.pop()
        n_split = str(last_noun).split()

        out_str[last_verb.i] = str(last_noun)
        if flag:
          out_str[last_noun.start] = str(last_verb)
          untouch[last_noun.start]  = str(last_noun)
        else:
          out_str[last_noun.i] = str(last_verb)
          untouch[last_noun.i]  = str(last_noun)


        untouch[last_verb.i] = str(last_verb)

    if len(verb_stack):
      for t in verb_stack:
        out_str[t.i] = str(t)
        untouch[t.i] = str(t)
    if len(noun_stack):
      for t in noun_stack:
        if flag:
          out_str[t.start] = str(t)
          untouch[t.start] = str(t)
        else:
          out_str[t.i] = str(t)
          untouch[t.i] = str(t)

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

def adverbVerbSwap(in_str, nlp = None):
    toks = nlp(in_str)
    s_l = [_ for _ in toks]
    assert len(s_l) > 5

    if toks[-1].pos_ != 'PUNCT':
      toks = nlp(in_str + ' .')
    verb_stack = []
    adverb_stack = []
    out_str = ['' for _ in range(len(toks))]
    for t in toks:
        if t.pos_ == 'VERB':
          verb_stack.append(t)
        if t.pos_ == 'ADV':
          adverb_stack.append(t)

    assert len(adverb_stack) >= 1 and len(verb_stack) >= 1

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

def nounAdjSwap(in_str, nlp = None):
    toks = nlp(in_str)
    s_l = [_ for _ in toks]
    assert len(s_l) > 5

    if toks[-1].pos_ != 'PUNCT':
      toks = nlp(in_str + ' .')
    noun_stack = []
    adj_stack = []
    out_str = ['' for _ in range(len(toks))]
    untouch = ['' for _ in range(len(toks))]
    for t in toks:
        if t.pos_ == 'ADJ':
            adj_stack.append(t)
        if t.pos_ in ['NOUN','PRON']:
          noun_stack.append(t)

    assert len(noun_stack) >= 1 and len(adj_stack) >= 1

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

def nounVerbSwap(in_str, nlp = None):
    toks = nlp(in_str)
    s_l = [_ for _ in toks]
    assert len(s_l) > 5

    if toks[-1].pos_ != 'PUNCT':
      toks = nlp(in_str + ' .')
    verb_stack = []
    noun_stack = []
    out_str = ['' for _ in range(len(toks))]
    untouch = ['' for _ in range(len(toks))]
    flag = True
    for t in toks:
        if t.pos_ in ['VERB','AUX']:
            verb_stack.append(t)
    try:
        for np in toks.noun_chunks:
            noun_stack.append(np)
    except:
        flag = False
        for t in toks:
            if t.pos_ in ['NOUN','PRON']:
              noun_stack.append(t)

    assert len(noun_stack) >= 1 and len(verb_stack) >= 1

    while len(verb_stack) and len(noun_stack):
        last_noun = noun_stack.pop(0)
        last_verb = verb_stack.pop(0)
        n_split = str(last_noun).split()

        out_str[last_verb.i] = str(last_noun)
        if flag:
          out_str[last_noun.start] = str(last_verb)
          untouch[last_noun.start]  = str(last_noun)
        else:
          out_str[last_noun.i] = str(last_verb)
          untouch[last_noun.i]  = str(last_noun)


        untouch[last_verb.i] = str(last_verb)

    if len(verb_stack):
      for t in verb_stack:
        out_str[t.i] = str(t)
        untouch[t.i] = str(t)
    if len(noun_stack):
      for t in noun_stack:
        if flag:
          out_str[t.start] = str(t)
          untouch[t.start] = str(t)
        else:
          out_str[t.i] = str(t)
          untouch[t.i] = str(t)
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


def verbSwaps(in_str, nlp = None):
    toks = nlp(in_str)
    s_l = [_ for _ in toks]
    assert len(s_l) > 5

    if toks[-1].pos_ != 'PUNCT':
      toks = nlp(in_str + ' .')
    verbs = []
    out_str = []
    for t in toks:
        if t.pos_ in ['VERB','AUX']:
            verbs+=[str(t)]

    verb_test = list(set(verbs))
    assert len(verb_test) > 1

    random.shuffle(verbs)
    verbs_gen = (_ for _ in verbs)

    for t in toks:
        if t.pos_ in ['VERB','AUX']:
            out_str += [next(verbs_gen)]
        else:
            out_str+= [str(t)]

    return ' '.join(out_str)


def wordShuffle(in_str, nlp = None):
    toks = nlp(in_str)

    if toks[-1].pos_ != 'PUNCT':
      toks = nlp(in_str + ' .')
    str_l = [str(_) for _ in toks]

    assert len(str_l) > 5

    random.shuffle(str_l[:-1])
    return ' '.join(str_l) + ' ' + str(str_l[-1])

def shuffleHalvesFirst(in_str, nlp = None):
    toks = nlp(in_str)
    if toks[-1].pos_ != 'PUNCT':
      toks = nlp(in_str + ' .')

    str_l = [str(_) for _ in toks]

    assert len(str_l) > 5

    first_half = str_l[:len(str_l)//2]

    random.shuffle(first_half)

    out_str = ' '.join(first_half) + ' '.join(str_l[len(str_l)//2:])

    return out_str

def shuffleHalvesLast(in_str, nlp = None):
    toks = nlp(in_str)
    if toks[-1].pos_ != 'PUNCT':
      toks = nlp(in_str + ' .')

    str_l = [str(_) for _ in toks]

    assert len(str_l) > 5
    random.seed(len(str_l))
    second_half = str_l[len(str_l)//2:-1]

    random.shuffle(second_half)

    out_str = ' '.join(str_l[:len(str_l)//2]) + ' '.join(second_half) + str_l[-1]

    return out_str

def verbAtBeginning(in_str, nlp = None):
  toks = nlp(in_str)
  if toks[-1].pos_ != 'PUNCT':
    toks = nlp(in_str + ' .')
  out_str = []
  flag = True

  for sent in toks.sents:
    flag = True
    temp_str = []
    pos = [t.pos_ for t in sent]
    assert 'VERB' in pos
    for t in sent:
        if flag:
            if t.pos_ == 'VERB':
                out_str = [str(t)] + out_str
                flag == False
                continue
        temp_str += [str(t)]
    out_str.append(' '.join(temp_str))
  return ' '.join(out_str)

def nounSwaps(in_str, nlp = None):
  toks = nlp(in_str)
  if toks[-1].pos_ != 'PUNCT':
    toks = nlp(in_str + ' .')
  noun_stack = []
  out_str = ['' for _ in range(len(toks))]
  untouch = ['' for _ in range(len(toks))]

  flag = True
  try:
      for np in toks.noun_chunks:
          noun_stack.append(np)
  except:
      flag = False
      for t in toks:
          if t.pos_ in ['NOUN','PRON']:
            noun_stack.append(t)

  ns_test = list(set(noun_stack))
  assert len(ns_test) > 1

  unshuffled = [_ for _ in noun_stack]
  random.shuffle(noun_stack)

  while len(noun_stack):
    noun_mis = noun_stack.pop(0)
    noun_real = unshuffled.pop(0)

    if flag:
      out_str[noun_real.start] = str(noun_mis)
      untouch[noun_real.start]  = str(noun_real)
    else:
      out_str[noun_real.i] = str(noun_mis)
      untouch[noun_real.i]  = str(noun_real)

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

def conjunctionShuffle(in_str, nlp = None):
  toks = nlp(in_str)
  if toks[-1].pos_ != 'PUNCT':
    toks = nlp(in_str + ' .')
  conjs = []
  out_str = []
  for t in toks:
      if 'CONJ' in str(t.pos_):
          conjs+=[str(t)]
  assert len(conjs) > 1

  for t in toks:
      if 'CONJ' in t.pos_:
          out_str += [conjs.pop()]
      else:
          out_str+= [str(t)]

  return ' '.join(out_str)
