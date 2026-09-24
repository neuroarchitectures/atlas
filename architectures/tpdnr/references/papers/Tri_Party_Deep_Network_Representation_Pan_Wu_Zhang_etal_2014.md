# Tri Party Deep Network Representation Pan Wu Zhang etal 2014

> Source: `Tri_Party_Deep_Network_Representation_Pan_Wu_Zhang_etal_2014.pdf`

---

                        Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence (IJCAI-16)




                                   Tri-Party Deep Network Representation
                Shirui Pan† , Jia Wu† , Xingquan Zhu⇤ , Chengqi Zhang† , Yang Wang‡
   †
     Centre for Quantum Computation & Intelligent System, FEIT, University of Technology Sydney
⇤
  Dept. of Computer & Electrical Engineering and Computer Science, Florida Atlantic University, USA
                             ‡
                               The University of New South Wales, Australia
      {shirui.pan, jia.wu, chengqi.zhang}@uts.edu.au; xzhu3@fau.edu; wangy@cse.unsw.edu.au

                          Abstract                                            Tang et al., 2015; Yang et al., 2015; Tian et al., 2014;
                                                                              Chang et al., 2015] encodes each node in a common, contin-
     Information network mining often requires exami-                         uous, and low-dimensional space while preserving the neigh-
     nation of linkage relationships between nodes for                        borhood relationship between nodes, so machine learning
     analysis. Recently, network representation has                           algorithms can be applied directly. For network represen-
     emerged to represent each node in a vector format,                       tation, existing methods mainly employ network structure
     embedding network structure, so off-the-shelf ma-                        based methods or node content based methods.
     chine learning methods can be directly applied for
                                                                                 Majority network representation methods are based on net-
     analysis. To date, existing methods only focus on
                                                                              work structure. Early works aim to learn social dimensions
     one aspect of node information and cannot lever-                         [Tang and Liu, 2009] or knowledge bases [Bordes et al.,
     age node labels. In this paper, we propose TriDNR,
                                                                              2011; Socher et al., 2013] to embed the entities in a network.
     a tri-party deep network representation model, us-
                                                                              Inspired by the deep learning techniques in natural language
     ing information from three parties: node struc-
                                                                              processing [Mikolov et al., 2013], Perozzi et al. [Perozzi
     ture, node content, and node labels (if available) to
                                                                              et al., 2014] recently proposed a DeepWalk algorithm which
     jointly learn optimal node representation. TriDNR
                                                                              employs neural network models to learn a feature vector for
     is based on our new coupled deep natural language
                                                                              each node from a corpus of random walks generated from
     module, whose learning is enforced at three levels:
                                                                              networks. These methods take the network structure as input
     (1) at the network structure level, TriDNR exploits
                                                                              but ignore content information associated to each node.
     inter-node relationship by maximizing the proba-
     bility of observing surrounding nodes given a node                          From the content perspective, many methods exist to rep-
     in random walks; (2) at the node content level,                          resent a text message into a vector space. Early studies em-
     TriDNR captures node-word correlation by maxi-                           ploy bag of word approaches (e.g., TFIDF) or topic models
     mizing the co-occurrence of word sequence given a                        (e.g. LDA [Blei et al., 2003]) to represent each document
     node; and (3) at the node label level, TriDNR mod-                       as a vector. However, these models do not consider the con-
     els label-word correspondence by maximizing the                          text information of a document (i.e., order of words), result-
     probability of word sequence given a class label.                        ing in suboptimal representation. Recently, neural network
     The tri-party information is jointly fed into the neu-                   or deep learning based approaches [Mikolov et al., 2013;
     ral network model to mutually enhance each other                         Le and Mikolov, 2014; Luong et al., 2013] have emerged for
     to learn optimal representation, and results in up                       text embedding. The Skip-gram model [Mikolov et al., 2013]
     to 79% classification accuracy gain, compared to                         employs a simple neural network model to learn distributed
     state-of-the-art methods.                                                vectors for words. Due to its simplicity, efficiency, and scala-
                                                                              bility, Skip-gram model is further extended as paragraph vec-
                                                                              tor model [Le and Mikolov, 2014], which learns latent vector
1   Introduction                                                              representation for arbitrary piece of text.
Many network applications, such as social networks, pro-                         The drawback of existing methods [Tang and Liu, 2009;
tein networks, and citation networks, are characterized with                  Perozzi et al., 2014; Tang et al., 2015; Le and Mikolov, 2014],
complex structure and rich node content information, where                    regardless of network exploration approaches or text model-
the network structure is naturally sparse in that only a small                ing methods, is twofold: (1) they only utilize one source of
number of nodes are connected and the node content is usu-                    information, so the representation is inherently shallow. (2)
ally represented as text information indicating the properties                all these methods learn network embedding in a fully unsu-
of a node, such as the title or abstract information of each                  pervised way. For many tasks, such as node classification, we
paper (node) in a citation network. The complexity of net-                    may have a limited number of labeled nodes, which provides
worked data imposes great challenge to many machine learn-                    usefully information to assist network representation.
ing tasks such as node classification in networks. To address                    When considering networks as a whole, the main challenge
this problem, network representation [Perozzi et al., 2014;                   of learning latent representation for network nodes is twofold:



                                                                       1895
  Network Structure
                             5
                                     15
                                                   7
                                                            8          9                                                                                       expensive matrix operation like SVD decomposition, which
          1                                                                12                3
                                                                                                      2
                                                                                                          5
                                                                                                                   16                                          prohibits TADW from handling large scale data.
                                                                                                                                                                  In this paper, we propose a tri-party deep network repre-
                                                       10                              15
                         4                6                                                               4
                                                                                                      1
    2                        14                                                             14
                                                                                13
               3                              16                11

                                                                                       10
                                                                                                 6
                                                                                                                                 7             9
                                                                                                                                                               sentation model, TriDNR, which uses a coupled neural net-
  Text    1 Transfer learning across networks for collective classification
                                                                                                                        8
                                                                                                                                          10
                                                                                                                                                               work architecture to exploit network information from three
          14   Domain transfer for video concept detection
                                                                                       5
                                                                                                               15
                                                                                                                            11
                                                                                                                                                   13
                                                                                                                                                               parties: node structure, node content, and node labels (if
                                                                                                                                                               available). At the node level, we maximize the probabil-
     9 A variable bit rate video codec for asynchronous transfer mode networks                                                            12


  Label            Class1: Labeled        Class2: Labeled            Unlabeled Nodes
                                                                                                 20           40                     60
                                                                                                                                                               ity of observing the surrounding nodes given a node vi in
                    (a) Input: A Network                                               (b) Output: Representation                                              a random walk sequence, which well preserves inter-node
                                                                                                                                                               relationship in networks. At the node content and node la-
Figure 1: Our model takes an information network as input                                                                                                      bel levels, we maximize the co-occurrence of word sequence
and outputs a low-dimensional representation for each node.                                                                                                    given a node and co-occurrence of word sequence given a la-
In this toy citation network, each node denotes a paper, links                                                                                                 bel, which accurately captures the node-word correlation and
are citation relationships, and texts are paper titles. Red and                                                                                                label-word correspondence. The tri-party information mutu-
blue nodes are labeled nodes from two classes (transfer learn-                                                                                                 ally enhances each other for optimal node representation. An
ing vs. video coding). Remaining nodes are unlabeled. (1)                                                                                                      example of using TriDNR to represent nodes of a citation net-
paper 1 and paper 9 share two common words transfer and                                                                                                        work to a two dimensional space is shown in Fig 1.
networks in their titles. If we ignore their labels, they will
likely be represented with similar representation. So label                                                                                                    2   Problem Definition
is useful. (2) paper 14 and paper 1 belong to the transfer
                                                                                                                                                               An information network is represented as G = (V, E, D, C),
learning class. Although paper 14 is more similar to paper 9
                                                                                                                                                               where V = {vi }i=1,··· ,N consists of a set of nodes, ei,j =
in terms of shared title words, the weak citations/connections
                                                                                                                                                               (vi , vj ) 2 E is an edge encoding the edge relationship be-
between paper 1 and paper 14 will make them close to each in
                                                                                                                                                               tween the nodes, di 2  S D is a text document associated with
the learned node representation space. So structure is useful
                                                                                                                                                               each node vi , C = L U is the class label information of the
for node representation. (3) By combining structure, text, and
                                                                                                                                                               network, with L denoting the labeled nodes and U being the
labels, our method is positioned to learn best representations.
                                                                                                                                                               unlabeled nodes. The network representation aims to learn a
                                                                                                                                                               low-dimensional vector vvi 2 Rk (k is a smaller number) for
  • Network Structure, Node Content, and Label Infor-                                                                                                          each node vi in the network, so that nodes close to each other
    mation Integration (Challenge 1) How to learn node                                                                                                         in network topology or with similar text content, or sharing
    embedding for networks containing links and rich text                                                                                                      the same class label information are close in the representa-
    information, and further tailor the representation for                                                                                                     tion space. Fig 1 demonstrates the importance of combining
    learning tasks provided with labeled samples?                                                                                                              structure, text, and labels for learning good representations.
  • Neural Network Modeling (Challenge 2) How to de-                                                                                                              In this paper, we assume the network has partial labeled
    sign effective neural network models for networked data,                                                                                                   nodes. If the label set L = ;, the representation becomes
    in order to obtain deep network representation results?                                                                                                    pure unsupervised, and our proposed solution is still valid.
   In terms of neural network modeling (Challenge 2), a                                                                                                        3   Preliminary: Skip-Gram and DeepWalk
straightforward method for this problem is to separately learn
a node vector by using DeepWalk [Perozzi et al., 2014] for                                                                                                     Recently, Skip-Gram [Mikolov et al., 2013], which is a lan-
network structure and learn a document vector via Paragraph                                                                                                    guage model exploiting word orders in a sequence and as-
Vectors model [Le and Mikolov, 2014] (the state-of-the art                                                                                                     suming that words closer are statistically more dependent or
model to embed text into a vector space), and then concate-                                                                                                    related, has drawn much attention due to its simplicity and ef-
nate these two vectors into a unified representation. How-                                                                                                     ficiency. Specifically, Skip-Gram aims to predict the context
ever this simple combination is suboptimal because it ignores                                                                                                  (surrounding words) within a certain window given current
the label information, and overlooks interactions between net-                                                                                                 word by maximizing the following log-likelihood:
work structures and text information.                                                                                                                                               T
                                                                                                                                                                                    X
   In order to exploit both structure and text information                                                                                                                     L=         log P(wt b : wt+b |wt )          (1)
for network representation (Challenge 1), a recent work                                                                                                                             t=1
TADW [Yang et al., 2015] shows that DeepWalk is equivalent                                                                                                     where b is the context width (window size), wt b : wt+b is a
to factorize a matrix M (sum of a series transition matrix).                                                                                                   sequence of words excluding the word wt itself. The proba-
However, this method has following drawbacks: (1) The ac-                                                                                                      bility P(wt b : wt+b |wt ) is computed as:
curate matrix M for factorization is non-trivial and very diffi-                                                                                                                       Y
cult to obtain. As a result, TADW has to factorize an approxi-                                                                                                                                   P(wt+j |wt )              (2)
mate matrix, which will weaken its representation power; (2)                                                                                                                        bjb,j6=0

TADW simply ignores the context of text information (e.g.,                                                                                                     Eq. (2) assumes that the contextual words wt b : wt+b are
orders of words), so cannot appropriately capture the seman-                                                                                                   independent given word wt . P(wt+j |wt ) is computed as:
tics of the words and nodes in a network setting; (3) unlike
                                                                                                                                                                                                  wt vwt+j )
                                                                                                                                                                                             exp(v>   0
neural network models that can scale up to millions of records
                                                                                                                                                                             P(wt+j |wt ) = PW                             (3)
in a single machine [Mikolov et al., 2013], TADW requires                                                                                                                                     w=1 exp(vwt vw )
                                                                                                                                                                                                        > 0




                                                                                                                                                        1896
where vw and v0w are the input and output vectors of w. After                         DeepWalk Model
                                                                                                                      Inter-Node Relationship Modeling
training the model, the input vector vw can be used as the final                                                       output       v3    v2        v4   v7
representation of word w.                                                                 v3    v2        v4   v7
                                                                                                                                                                TriDNR Model
                                                                             output
                                                                                                                       projection
3.1     DeepWalk Model
                                                                             projection
                                                                                                                                                                       w5
Motivated by the Skip-Gram, DeepWalk [Perozzi et al., 2014]                                                            input
                                                                                                                                               v1
                                                                                                                                                                       w3
constructs a corpus S that consists of random walks generated                input                   v1
                                                                                                                                               c1                      w2
from the network. Each random walk v1 ! v3 ! · · · ! vn                                                                                    input         projection   output
can be considered as a sentence and each node vi can be con-                                                                                   Node-word & Label-word
sidered as a word in neural language models. Then Deep-                                                                                         Correlation Modeling

Walk algorithm trains the Skip-Gram model [Mikolov et al.,
2013] on random walks, and obtain a distributed representa-                 Figure 2: Architecture of the DeepWalk (Skip-Gram) method
tion vector for each node. The objective of DeepWalk model                  and our proposed TriDNR method. The DeepWalk approach
is to maximize the likelihood of the surrounding nodes given                learns the network representation based on the network struc-
current node vi for all random walks s 2 S:                                 ture only. Our TriDNR method couples two neural networks
                    N X
                                                                            to learn the representation from three parties (i.e., node struc-
                    X                                                       ture, node content, and node label) to capture the inter-node,
              L=              log P(vi b : vi+b |vi )
                    i=1 s2S
                                                                            node-word, and label-word relationship. The input, projec-
                    N X
                                                               (4)          tion, and output indicate the input layer, hidden layer, and
                    X             X                                         output layer of a neural network model.
                =                           log P(vi+j |vi )
                    i=1 s2S    bjb,j6=0

where N is the total number of nodes. The architecture of
                                                                             2. Node-content correlations Assessing. The lower layer
DeepWalk model is illustrated in the left panel of Fig 2.
                                                                                of TriDNR models the contextual information (node-
  DeepWalk can only utilize the network information for
                                                                                content correlations) of words within a document.
model learning, without considering any text information
augmented with each node and label information provided by
the specific task like node classification.                                  3. Connections. We couple these two layers by the node
                                                                                v1 in the model, indicating that v1 is influenced by both
                                                                                random walk sequences and node content information.
4     Tri-Party Deep Network Representation
In this section, we present our tri-party deep network rep-                  4. Label-content Correspondence Modeling To utilize
resentation algorithm for jointly utilizing network structure,                  the valuable class label information of each node, we
text content, and label information to learn a latent vector for                also use the label of each document as input and simul-
each node in the network.                                                       taneously learn the input label vectors and output word
   We use neural network models to learn vvi , the latent repre-                vectors, providing directly modeling between node la-
sentation for node vi in a network. This latent representation                  bels and node content.
vvi acts as an input vector in our neural network model. More
specifically, our TriDNR algorithm consists of two steps:                   Note that the label information is not used for inter-node rela-
    1. Random Walk Sequence Generation uses network                         tionship modeling. This is because we can hardly assess the
       structure as the input and randomly generates a set of               class label of a random walk sequence.
       walks over the nodes, with each walk rooting at a node                 The lower level (panel) of Fig 2, which exploits document
       vi and randomly jumping to other nodes each time. The                and class label information, can be formalized by the follow-
       random walk corpus can capture the node relationship.                ing objective function:
    2. Coupled Neural Network Model Learning embeds
       each node into a continuous space by considering the                                    |L|                              N
                                                                                               X                                X
       following information: (a) random walk corpus which                           L=               log P(w b : wb |ci ) +             log P(w b : wb |vi )                  (5)
       captures the inter-node relationship, (b) the text corpus                               i=1                              i=1
       which models the node-content correlations, (c) label in-
       formation which encodes label-node correspondences.
                                                                            where w b : wb is a sequence of words inside a contextual
                                                                            window of length b. ci is the class label of node vi . Note that
4.1     Model Architecture                                                  if no label information is used (i.e., |L| = 0), the first term
Our coupled neural network model architecture is illustrated                vanishes, Eq. (5) will become the Paragraph Vector model
in the right panel of Fig 2, which has the following properties:            [Le and Mikolov, 2014], which exploits the text information
    1. Inter-node Relationship Modeling. The upper layer of                 to learn vector representation for each document.
       TriDNR learns the structure relationship from the ran-                 Given a network G which consists of N nodes (i.e., V =
       dom walk sequences, under assumption that connected                  {vi }i=1,··· ,N ), suppose the random walks generated in G is
       nodes are statistically dependent on each other.                     S, our TriDNR model aims to maximize the following log-



                                                                     1897
likelihood:                                                                               dimension as the leaves. So instead of enumerating all nodes
                      N
                      XX          X                                                       in Eq. (7) in each gradient step, we only need to evaluate the
  L = (1         ↵)                          log P(vi+j |vi )                             path from the root to the corresponding leaf in the Huffman
                      i=1 s2S   bjb,j6=0                                                trees. Suppose the path to the leaf node vi is a sequence of
           N                                    |L|                                       vertices (l0 , l1 , · · · , lP ), (l0 = root, lP = vi ), then
           X X                                  X   X
      +↵                  log P(wj |vi ) + ↵                    log P(wj |ci )                                                    P
                                                                                                                                  Y
           i=1    bjb                         i=1   bjb                                                      P(vi+j |vi ) =         P(lt |vi )      (10)
                                                                             (6)                                                  t=1
where ↵ is the weight that balances network structure, text,                              where P(lt |vi ) can be further modeled by a binary classifier
and label information. b is the window size of sequence, and                              which is defined as:
wj indicates the j-th word in a contextual window.
   Given equation Eq. (6), the first term requires the calcula-                                                    P(lt |vi ) = (v>   0
                                                                                                                                  vi vvlt )             (11)
tion of P(vi+j |vi ), the probability of observing surrounding
                                                                                          here (x) is the sigmoid function, and v0vlt is the representa-
nodes given current node vi , which can be computed using
the soft-max function as follows:                                                         tion of tree vertex lt ’s parent. By doing this, the time com-
                                                                                          plexity is reduced to O(N log N ). Similarly, we can use hier-
                                      exp(v>vi vvi+j )
                                                0
                      P(vi+j |vi ) = PN                                      (7)          archical soft-max technique to compute the words and labels
                                       v=1 exp(vvi vv )
                                                  > 0                                     in Eq. (8) and Eq. (9).
where vv and v0v are the input and output vector represen-                                5       Experimental Results
tation of node v. Furthermore, the probability of observing
contextual words w b : wb given current node vi is:                                       5.1       Experimental Setup
                                                                                          We report our experimental results on two networks. Both of
                                          vi vwj )
                                     exp(v>   0
                       P(wj |vi ) = PW                                       (8)          them are citation networks and we use the paper title as node
                                     w=1 exp(vvi vw )
                                                > 0
                                                                                          content for each node in the networks.
where v0wj is the output representation of word wj and W is                                  DBLP dataset 1 consists of bibliography data in computer
                                                                                          science [Tang et al., 2008]. Each paper may cite or be cited
the number of distinct words in the whole network. Similarly,
                                                                                          by other papers, which naturally forms a citation network.
the probability of observing the words given a class label ci
                                                                                          In our experiments, we select a list of conferences from 4
is defined as:
                                                                                          research areas, database (SIGMOD, ICDE, VLDB, EDBT,
                                          ci v w j )
                                     exp(v>    0
                                                                                          PODS, ICDT, DASFAA, SSDBM, CIKM), data mining
                       P(wj |ci ) = PW                                       (9)          (KDD, ICDM, SDM, PKDD, PAKDD), artificial intelligent
                                     w=1 exp(vci vw )
                                                 > 0
                                                                                          (IJCAI, AAAI, NIPS, ICML, ECML, ACML, IJCNN, UAI,
   From Eq. (8) and Eq. (9), we know that the text informa-                               ECAI,COLT, ACL, KR), computer vision (CVPR, ICCV,
tion and label information will jointly affect v0wj , the output                          ECCV, ACCV, MM, ICPR, ICIP, ICME). The DBLP network
representation of word wj , which will further propagate back                             consist of 60,744 papers (nodes), 52,890 edges in total.
to influence the input representation of vi 2 V in the network.                              CiteSeer-M10 is a subset [Lim and Buntine, 2014] of
As a result, the node representation (i.e., the input vectors of                          CiteSeerX data 2 which consist of scientific publications from
nodes) will be enhanced by both network structure, text con-                              10 distinct research areas: agriculture, archaeology, biology,
tent, and label information.                                                              computer science, financial economics, industrial engineer-
                                                                                          ing, material science, petroleum chemistry, physics, and so-
4.2   Model Optimization                                                                  cial science. This dataset consists of multidisciplinary 10
We train our model Eq. (6) using stochastic gradient ascent,                              classes, with 10,310 publications and 77,218 edges in total.
which is suitable for large-scale data. However, computing                                   The following algorithms are comparing in our paper:
the gradient in Eq. (7) is expensive, as it is proportional to                              1. DeepWalk [Perozzi et al., 2014] learns network repre-
the number of nodes in the network N . Similarly, comput-                                      sentation using network structure only.
ing the gradient in Eq. (8) and Eq. (9) are proportional to
the unique number of words in the document content W . To                                   2. LINE [Tang et al., 2015] is a state-of-the-art algorithm
handle this problem, we resort to the hierarchical soft-max                                    for network representation based on network structure.
[Morin and Bengio, 2005], which reduces the time complex-                                   3. Doc2Vec [Le and Mikolov, 2014] is the Paragraph Vec-
ity to O(Rlog (W ) + N log (N )) where R is the total number                                   tors algorithm which embeds any piece of text in a dis-
of words in the document content.                                                              tributed vector using neural network models. Here we
   Specifically, the hierarchical model in our algorithm uses                                  use PV-DBOW model in [Le and Mikolov, 2014].
two binary trees, one with distinct nodes as leaves and another                             4. LDA [Blei et al., 2003] is the Latent Dirichlet alloca-
with distinct words as leaves. The tree is built using Huffman                                 tion algorithm that learns a topic distribution to represent
algorithm so that each vertex in the tree has a binary code and                                each document (or text).
more frequent nodes (or words) has shorter codes. There is a
                                                                                              1
unique path from the root to each leaf. The interval vertices of                                  http://arnetminer.org/citation (V4 version is used)
                                                                                              2
the trees are represented as real-valued vector with the same                                     http://citeseerx.ist.psu.edu/




                                                                                   1898
                     Table 1: Average Macro F1 Score and Standard Deviation on Citeseer-M10 Network

%p     DeepWalk       Doc2Vec       DW+D2V          DNRL               LDA                                     RTM                                   LINE                                      TADW                                TriDNR
 10    0.354±0.007   0.432±0.008    0.495±0.007   0.564±0.007       0.458±0.005                        0.504±0.007                          0.531±0.004                                  0.600±0.008                            0.626±0.009
 30    0.411±0.002   0.477±0.005    0.586±0.004   0.667±0.003       0.549±0.003                        0.598±0.004                          0.569±0.006                                  0.652±0.005                            0.715±0.004
 50    0.425±0.004   0.494±0.004    0.614±0.006   0.702±0.006       0.581±0.009                        0.629±0.005                          0.581±0.005                                  0.671±0.006                            0.753±0.006
 70    0.434±0.007   0.503±0.006    0.628±0.009   0.725±0.005       0.589±0.007                        0.638±0.011                          0.589±0.010                                  0.681±0.005                            0.777±0.006

                         Table 2: Average Macro F1 Score and Standard Deviation on DBLP Network

%p     DeepWalk       Doc2Vec       DW+D2V          DNRL               LDA                                     RTM                                   LINE                                      TADW                                TriDNR
 10    0.398±0.004   0.605±0.005    0.653±0.005   0.663±0.002       0.644±0.002                        0.665±0.004                          0.427±0.003                                  0.676±0.006                            0.687±0.004
 30    0.423±0.003   0.617±0.003    0.681±0.003   0.698±0.002       0.652±0.003                        0.674±0.001                          0.438±0.003                                  0.689±0.004                            0.727±0.002
 50    0.426±0.002   0.620±0.003    0.686±0.003   0.710±0.002       0.655±0.003                        0.676±0.002                          0.438±0.002                                  0.692±0.003                            0.738±0.003
 70    0.428±0.004   0.623±0.003    0.690±0.003   0.718±0.003       0.654±0.003                        0.678±0.003                          0.439±0.003                                  0.695±0.003                            0.744±0.002


  5. RTM [Chang and Blei, 2009] is the relational topic
                                                                                              Macro F1 on DBLP Dataset w.r.t Different Number of Features                           Macro F1 on Citeseer−M10 Dataset w.r.t Different Number of Features


                                                                                       0.8                                                                                   0.8

      model which captures both text and network structure
      to learn topic distributions of each document.                                   0.7                                                                                   0.7




  6. TADW [Yang et al., 2015] is a state-of-the-art algorithm               Macro F1                                                                              Macro F1
                                                                                       0.6                                                                                   0.6

                                                                                                                                                     DeepWalk                                                                                  DeepWalk


      that utilizes both network and text information to learn
                                                                                                                                                     Doc2Vec                                                                                   Doc2Vec
                                                                                       0.5                                                           DW+D2V                  0.5                                                               DW+D2V
                                                                                                                                                     DNRL                                                                                      DNRL


      the network representation.
                                                                                                                                                     LDA                                                                                       LDA
                                                                                       0.4                                                           RTM                     0.4                                                               RTM
                                                                                                                                                     LINE                                                                                      LINE
                                                                                                                                                     TADW                                                                                      TADW


  7. DW+D2V approach simply concatenates the vector rep-
                                                                                                                                                     TriDNR                                                                                    TriDNR
                                                                                       0.3                                                                                   0.3
                                                                                         50         100          150           200             250          300                50            100           150           200             250              300
                                                                                                                 Number of Features                                                                        Number of Features

      resentations learned by different neural network models,
      i.e., DeepWalk and Doc2Vec.                                                       Figure 3: Results with Different Number of Features k.
  8. TriDNR is our proposed tri-party deep network repre-
      sentation algorithm that exploits network structure, node               Table 1 and Table 2 show that DeepWalk and LINE al-
      content, label information for learning.                             gorithms perform fairly poor on citation networks. This is
  9. DNRL is a variant of our TriDNR, which only uses node                 mainly because the network structure is rather sparse and only
      content and label information (lower layer of Fig 2) and             contains limited information. Doc2Vec (Paragraph Vectors)
      ignores the network structure information.                           is much better than DeepWalk, especially on DBLP dataset,
   Measures & Parameter Setting We perform node clas-                      as the text content (title) has rich information comparing to
sification to evaluate the quality of different algorithms. In             network structure. When concatenating the embeddings from
each network, p% nodes are randomly labeled, the rest are                  DeepWalk and Doc2Vec, as DW+D2V does, the representa-
unlabeled. The whole network, including node content, are                  tion is substantially improved. However, DW+D2V is still far
used to learn the network representation. Once we obtained                 from optimal, comparing with TriDNR algorithm.
the node vectors using different comparing methods, we train                  Importance of Label Information. Our experimental
a linear SVM from the training data (nodes) to predict unla-               results show that DNRL, which uses label information to
beled nodes. We choose a linear classifier, instead of non-                learn the network embedding (only used the lower layer
linear model or sophisticated relational classifiers [Sen et al.,          of our TriDNR model), has already significantly outper-
2008], in order to reduce the impact of complicated learning               formed the naive combination of two neural network mod-
approaches on the classification performance.                              els (DW+D2V). This observation validates the importance of
   The default parameter for TriDNR are set as follows: win-               label information for network representation.
dow size b=8, dimensions k =300, training size p = 30%,                       Effectiveness of Coupled design. When we add an-
↵ = 0.8. For fairness of comparison, all comparing algo-                   other layer of neural network on top of DNRL, our pro-
rithms will use the same same number of features k. The pa-                posed TriDNR model shows that it outperforms DeepWalk,
rameters for other algorithms will keep the same or be close               Doc2Vec, and DW+D2V by a large margin. The experimen-
to TriDNR as much as possible. For instance, for all neural                tal result demonstrated the importance of using label informa-
network models, we use window size b = 8. The rest param-                  tion and the effectiveness of our coupled architecture design.
eters are set following the suggestion in their original papers.              Topic model based algorithms: As for the topic model
For each parameter setting, we repeat the experiment 10 times              based algorithms, relational topic model (RTM) outperforms
and report the average results and standard deviation.                     the traditional topic model LDA. This is because RTM takes
                                                                           the network structure into consideration. Both LDA and RTM
5.2   Performance on Node Classification                                   are not comparable to TriDNR.
We vary the percentages of training samples p% from 10% to                    TriDNR vs. TADW: The results in Table 1 and 2 demon-
70% and report the results in Table 1 and Table 2.                         strate that TriDNR is also superior to TADW. This is be-



                                                                    1899
                            Figure 4: 2D visualization on Citeseer-M10 network by different algorithms.


cause: (1) TADW is a matrix factorization algorithm which
factorizes a matrix M (sum of a series transition matrix) to-
gether with a text matrix T . In reality, an accurate M is very
computationally expensive so TADW only factorizes an ap-
proximate M , which will affect its representation power; (2)
TADW does not consider context information, and its T ma-
trix cannot preserve the orders of word; (3) TADW does not
explore the valuable label information for a specific task, re-
sulting in suboptimal result only. In contrast, TriDNR takes
each document (node) and the words within of a window (in-
cluding the order information) into consideration, which well
preserve the neighborhood relationship and documents. Ad-                 Table 3: Top-5 Similar Node Search: matched results are (•)
ditionally, TriDNR exploits label information, which signifi-
cantly improves the representation power, leading to a good
                                                                           Query: Training Linear SVMs in Linear Time [Joachims, 2006]
classification performance.
   Overall, TriDNR significantly outperforms its peers. With               TriDNR :
                                                                           1. optimized cutting plane algorithm for support vector machines (•)
p%=70, TriDNR (0.777) beats DeepWalk (0.434) and TADW
                                                                           2. proximal regularization for online and batch learning (•)
(0.681) by 79.0% and 14.1% in the Citeseer-M10 network.                    3. a sequential dual method for large scale multi-class linear svms (•)
                                                                           4. a dual coordinate descent method for large-scale linear svm (•)
Results with different number of dimensions k                              5. bootstrap based pattern selection for support vector regression (•)
We vary the number of dimensions (k) used in comparing al-
                                                                           Doc2Vec :
gorithms, and report the results in Fig 3. When k increases                1. training linear discriminant analysis in linear time (•)
from 50 to 150, there is a slightly increase for TriDNR al-                2. circularity measuring in linear time
gorithm. Afterwards, not much difference is observed with                  3. mining association rules in hypertext databases
different number of features. The results show that TriDNR                 4. approximate matching in xml
is very stable with various dimension size.                                5. disaggregations in databases
                                                                           DeepWalk :
5.3      Case Study                                                        1. bootstrap based pattern selection for support vector regression (•)
                                                                           2. large margin training for hidden markov models with ...
In this subsection, we retrieve the most similar nodes w.r.t. a            3. optimized cutting plane algorithm for support vector machines (•)
given query node. Specifically, we compute the top 5 nearest               4. proximal regularization for online and batch learning (•)
neighbors with Consine distance based on the vector repre-                 5. extremely fast text feature extraction for classification and indexing
sentations learned by different algorithms. The results are                TADW :
shown in Table 3.                                                          1. Circularity Measuring in Linear Time
   The query paper is the best paper of KDD 2006 and is also               2. Training Linear Discriminant Analysis in Linear Time (•)
one of the top-10 downloaded paper according to ACM dig-                   3. Maximal Incrementality in Linear Categorial Deduction
ital library 3 (access on 19/01/2016). Basically, the paper                4. Variational Linear Response
                                                                           5. How Linear are Auditory Cortical Responses
considers using advanced optimization techniques like cut-
ting plane algorithm for training linear svm algorithm.
   Table 3 shows that all results retrieved by TriDNR are sim-
ilar or closely related to SVM or optimization techniques. For
instance, the first result optimized cutting plane algorithm for
support vector machines is actually a following-up work of
the query paper. In contrast, the results returned by Doc2Vec
or TADW only have 1 record matching with the query, which
is mostly based on the term linear time. For DeepWalk, there
are indeed some similar answers by using the network infor-
mation (citation relationship), because this is a highly cited
   3
       http://dl.acm.org/event.cfm?id=RE329




                                                                   1900
paper (1348 according to Google Scholar by 19/01/2016).                   [Le and Mikolov, 2014] Quoc V Le and Tomas Mikolov. Dis-
However, its result are still worse than our TriDNR algorithm.               tributed representations of sentences and documents. In In-
                                                                             ternational Conference on Machine Learning, pages 1188–
5.4    Network Visualization                                                 1196, 2014.
An important application of network representation is to cre-             [Lim and Buntine, 2014] Kar Wai Lim and Wray Buntine.
ate meaningful visualizations that layout a network in two                   Bibliographic analysis with the citation network topic
dimensional space. Following [Tang et al., 2015], we learn                   model. In Proceedings of the Sixth Asian Conference on
a low-dimensional representation for each node and map the                   Machine Learning, pages 142–158, 2014.
Citeseer-10 network into a 2D space in Fig 4.                             [Luong et al., 2013] Minh-Thang Luong, Richard Socher, and
   In Fig 4, the visualization using Doc2Vec and TADW algo-                  Christopher D Manning. Better word representations with
rithms are not very meaningful, in which the papers from the                 recursive neural networks for morphology. CoNLL-2013,
same cluster are not clustered together. The result obtained by              104, 2013.
DeepWalk is slightly better, in which a large group is formed             [Mikolov et al., 2013] Tomas Mikolov, Kai Chen, Greg Cor-
representing the research of Petroleum Chemistry. However                    rado, and Jeffrey Dean. Efficient estimation of word repre-
papers from other groups are still highly overlapping. The vi-               sentations in vector space. arXiv preprint arXiv:1301.3781,
sualization results of TriDNR are quite clear, with meaningful               2013.
layout for each class.                                                    [Morin and Bengio, 2005] Frederic Morin and Yoshua Bengio.
                                                                             Hierarchical probabilistic neural network language model.
6     Conclusion                                                             In Proceedings of the international workshop on artificial
                                                                             intelligence and statistics, pages 246–252. Citeseer, 2005.
In this paper, we proposed a Tri-party deep network represen-
tation algorithm. We argued that most exiting algorithms are              [Perozzi et al., 2014] Bryan Perozzi, Rami Al-Rfou, and
simple shallow methods that only use one aspect of node in-                  Steven Skiena. Deepwalk: Online learning of social repre-
formation. In addition, none of the existing method is able to               sentations. In Proceedings of the 20th ACM SIGKDD inter-
utilize the label information in the network for representation.             national conference on Knowledge discovery and data min-
Accordingly, we proposed a coupled neural network based al-                  ing, pages 701–710. ACM, 2014.
gorithm to exploit inter-node relationships, node-content cor-            [Sen et al., 2008] Prithviraj Sen, Galileo Namata, Mustafa Bil-
relation, and label-content correspondence in a networks to                  gic, Lise Getoor, Brian Galligher, and Tina Eliassi-Rad. Col-
learn an optimal representation for each node in the network.                lective classification in network data. AI magazine, 29(3):93,
Experimental results demonstrated the superb performance of                  2008.
our algorithms. The key contribution of the paper is twofold:
                                                                          [Socher et al., 2013] Richard Socher, Danqi Chen, Christo-
(1) we exploit network representation from multiple parties
and different network levels, and (2) we propose a new neu-                  pher D Manning, and Andrew Ng. Reasoning with neural
ral network model for deep network representation learning.                  tensor networks for knowledge base completion. In Ad-
                                                                             vances in Neural Information Processing Systems, pages
                                                                             926–934, 2013.
References                                                                [Tang and Liu, 2009] Lei Tang and Huan Liu. Relational
[Blei et al., 2003] David M Blei, Andrew Y Ng, and Michael I                 learning via latent social dimensions. In Proceedings of the
   Jordan. Latent dirichlet allocation. the Journal of machine               15th ACM SIGKDD international conference on Knowledge
   Learning research, 3:993–1022, 2003.                                      discovery and data mining, pages 817–826. ACM, 2009.
[Bordes et al., 2011] Antoine Bordes, Jason Weston, Ronan                 [Tang et al., 2008] Jie Tang, Jing Zhang, Limin Yao, Juanzi Li,
  Collobert, and Yoshua Bengio. Learning structured embed-                   Li Zhang, and Zhong Su. Arnetminer: extraction and min-
  dings of knowledge bases. In Twenty-Fifth AAAI Conference                  ing of academic social networks. In Proceedings of the 14th
  on Artificial Intelligence, 2011.                                          ACM SIGKDD international conference on Knowledge dis-
                                                                             covery and data mining, pages 990–998. ACM, 2008.
[Chang and Blei, 2009] Jonathan Chang and David M Blei.
  Relational topic models for document networks. In Inter-                [Tang et al., 2015] Jian Tang, Meng Qu, Mingzhe Wang, Ming
  national Conference on Artificial Intelligence and Statistics,             Zhang, Jun Yan, and Qiaozhu Mei. Line: Large-scale in-
  pages 81–88, 2009.                                                         formation network embedding. In Proceedings of the 24th
                                                                             International Conference on World Wide Web, pages 1067–
[Chang et al., 2015] Shiyu Chang, Wei Han, Jiliang Tang,                     1077. International World Wide Web Conferences Steering
  Guo-Jun Qi, Charu C Aggarwal, and Thomas S Huang. Het-                     Committee, 2015.
  erogeneous network embedding via deep architectures. In                 [Tian et al., 2014] Fei Tian, Bin Gao, Qing Cui, Enhong Chen,
  Proceedings of the 21th ACM SIGKDD International Con-
                                                                             and Tie-Yan Liu. Learning deep representations for graph
  ference on Knowledge Discovery and Data Mining, pages
                                                                             clustering. In AAAI, pages 1293–1299, 2014.
  119–128. ACM, 2015.
                                                                          [Yang et al., 2015] Cheng Yang, Zhiyuan Liu, Deli Zhao,
[Joachims, 2006] Thorsten Joachims. Training linear svms in
                                                                             Maosong Sun, and Edward Y Chang. Network represen-
   linear time. In Proceedings of the 12th ACM SIGKDD inter-                 tation learning with rich text information. In International
   national conference on Knowledge discovery and data min-                  Joint Conference on Artificial Intelligence, 2015.
   ing, pages 217–226. ACM, 2006.



                                                                   1901

