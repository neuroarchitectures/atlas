# A ributed Signed Network Embedding Wang Aggarwal Tang etal 2017

> Source: `A_ributed_Signed_Network_Embedding_Wang_Aggarwal_Tang_etal_2017.pdf`

---

                                     Attributed Signed Network Embedding
                                    Suhang Wang                                                                   Charu Aggarwal
                              Arizona State University                                                    IBM T. J. Watson Research Center
                              suhang.wang@asu.edu                                                                charu@us.ibm.com

                                     Jiliang Tang                                                                      Huan Liu
                            Michigan State University                                                          Arizona State University
                             tangjili@cse.msu.edu                                                                 huan.liu@asu.edu

ABSTRACT                                                                                      embedding is one of the most central tasks in data mining, and it
The major task of network embedding is to learn low-dimensional                               has been proven to be useful in many social network mining tasks
vector representations of social-network nodes. It facilitates many                           such as link prediction [18], community detection [21], node classi-
analytical tasks such as link prediction and node clustering and                              fication [4] and visualization [32]. The majority of existing network
thus has attracted increasing attention. The majority of existing                             embedding algorithms have been dedicated to social networks with
embedding algorithms are designed for unsigned social networks.                               only positive links. However, social networks can contain both
However, many social media networks have both positive and neg-                               positive and negative links, and these signed social networks are
ative links, for which unsigned algorithms have little utility. Recent                        present on a variety of social media sites, such as Epinions with
findings in signed network analysis suggest that negative links                               trust and distrust links, and Slashdot with friend and foe links. In
have distinct properties and added value over positive links. This                            addition to existing signed social networks, many algorithms are
brings about both challenges and opportunities for signed network                             proposed to construct signed networks from positive and negative
embedding. In addition, user attributes, which encode properties                              interactions between users or documents [9, 19].
and interests of users, provide complementary information to net-                                The availability of negative links in signed networks causes prob-
work structures and have the potential to improve signed network                              lems in leveraging the basic principles that are commonly used for
embedding. Therefore, in this paper, we study the novel problem                               mining unsigned social networks. This is because the principles
of signed social network embedding with attributes. We propose a                              of mining signed social networks can be substantially different
novel framework SNEA, which exploits the network structure and                                from those of unsigned networks [17, 28]. For example, homophily
user attributes simultaneously for network representation learning.                           effects and social influence for unsigned networks may not be appli-
Experimental results on link prediction and node clustering with                              cable to signed networks in their original form [30]. These present
real-world datasets demonstrate the effectiveness of SNEA.                                    challenges in extending existing algorithms from the unsigned case.
                                                                                              In a similar vein, signed network embedding cannot be easily car-
CCS CONCEPTS                                                                                  ried out by simply extending the existing embedding algorithms for
                                                                                              unsigned social networks. Recent research on mining signed social
•Information systems →Data mining;
                                                                                              networks suggests that negative links have added value over posi-
KEYWORDS                                                                                      tive links in various analytical tasks. For example, a small number
                                                                                              of negative links can significantly improve positive link prediction
Signed Social Networks, Network Embedding, Node Attributes                                    performance [16], and they can also improve recommendation per-
ACM Reference format:                                                                         formance in social media [33]. While signed network embedding is
Suhang Wang, Charu Aggarwal, Jiliang Tang, and Huan Liu. 2017. Attrib-                        challenging, its research results can potentially advance signed net-
uted Signed Network Embedding. In Proceedings of CIKM’17, November                            work mining tasks such as link prediction. However, the existing
6–10, 2017, Singapore., , 10 pages.                                                           work on signed network embedding is rather limited. In addition,
DOI: https://doi.org/10.1145/3132847.3132905
                                                                                              node attributes, which reveal users interests and/or properties, have
                                                                                              been proven to be effective for learning better representations for
1     INTRODUCTION                                                                            unsigned networks [7, 46]. Thus, we are curious if node attributes
The increasing availability of large-scale social media networks                              can help to improve the quality of signed network embedding.
has greatly advanced the ability to perform various mining tasks.                                In this paper, we investigate the novel problem of signed net-
An important task is that of network embedding, which aims at                                 work embedding with attributes in social media by studying the
learning low-dimensional vector representations of nodes. Network                             following two questions: (1) What’s the relationship between user
                                                                                              links and user attributes; and (2) how to model signed links and
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed         attributes simultaneously for learning network embeddings. To
for profit or commercial advantage and that copies bear this notice and the full citation     answer these two questions, we conduct data analysis on signed
on the first page. Copyrights for components of this work owned by others than ACM            network with attributes and based on the findings of the analysis
must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,
to post on servers or to redistribute to lists, requires prior specific permission and/or a   we propose a novel framework for Signed N etwork Embedding
fee. Request permissions from permissions@acm.org.                                            with Attributes, SNEA, which models both signed links and user
CIKM’17, November 6–10, 2017, Singapore.
                                                                                              attributes in a unified framework. The major contributions of the
© 2017 ACM. ISBN 978-1-4503-4918-5/17/11. . . $15.00
DOI: https://doi.org/10.1145/3132847.3132905                                                  paper are summarized next:
      • Providing a principled way to analyze the relationship           extending embedding algorithms for unsigned social networks. In
        between signed links and the similarity of user attributes;      addition, negative links have added value over positive links in var-
      • Proposing a novel framework SNEA, which leverages both           ious analytical tasks. For example, a small number of negative links
        signed links and user attributes for learning network em-        can significantly improve positive link prediction performance [16],
        bedding; and                                                     and they can also improve recommendation performance in social
      • Conducting experiments on real-world signed social net-          media [33]. Therefore, recently, signed network representation
        work datasets to assess the effectiveness of SNEA.               is attracting increasing attention [12, 16, 27, 45, 47]. In [16, 35],
                                                                         degree-based features such as the number of incoming positive and
   The remainder of the paper is organized as follows. In Section
                                                                         negative links of a node and triad based features that include the
2, we review related work. In Section 3, we give a preliminary
                                                                         structure information of a triad are defined manually and extracted
analysis of signed social networks with attributes, which lays the
                                                                         from the network to represent the nodes for sign prediction in net-
groundwork for SNEA. In Section 4, we introduce the details of
                                                                         works. Another work in [12] models signed networks using matrix
the proposed framework SNEA. In Section 5, we present a method
                                                                         factorization. Zheng et al. [47] extend the spectral embedding to
to solve the optimization problem of SNEA along with the time
                                                                         tackle signed network. Song et al. [27] propose two lower bounds
complexity analysis. In Section 6, we show empirical evaluation
                                                                         of GAUC to put more emphasis on ranking positive links on the
with discussion. In Section 7, we give conclusion with future work.
                                                                         top and negative links at the bottom of a ranking list. Wang et
                                                                         al. [36] propose a deep network based model for signed network
2   RELATED WORK                                                         embedding. As node attributes is helpful for unsigned network
Network embedding aims at learning low-dimensional vector rep-           work embedding, it has potential to improve the quality of signed
resentations for nodes of a given network. It has been proven to         network embedding. However, to the best of our knowledge, there’s
be useful in many tasks of network analysis such as link predic-         no existing work that exploits node attributes for signed link pre-
tion [18], community detection [5, 21], node classification [4, 37]      diction. Thus, in this paper, we study the novel problem of signed
and visualization [32]. The heterogeneity in data representation, the    network embedding with attributes. In particular, we propose a
sparsity of the network, and the varying degrees of various nodes,       novel framework SNEA, which leverages signed social network and
all play a significant role in making network mining tasks more          user attributes for learning better network representations.
challenging. To address the sparsity issue, network embedding en-
codes and represents each node in a unified low-dimensional space,       3    PROBLEM STATEMENT
which facilitates a better understanding of semantic relationships       Let U = {u 1 , u 2 , . . . , un } be a set of users where n is the number
and further alleviates the inconveniences caused by sparsity [22].       of users. A user ui can have positive or negative links to other
Network embedding has attracted increasing attention in recent           users, which results in a signed social network. Let G = {U, E}
years. The majority of them focuses on unsigned network em-              denote the signed social network where E ⊂ U × U is a set of
bedding [2, 10, 18, 20, 22, 24, 31, 32, 34, 39]. For example, in [2],    edges. We use A ∈ Rn×n to denote the adjacency matrix, where
spectral analysis is performed on Laplacian matrix and the top-k         Ai j = 1 means positive link from ui to u j , Ai j = −1 denotes neg-
eigenvectors are used as the representations of network nodes. t-        ative link and Ai j = 0 means the link is missing. Generally, links
SNE proposed in [32] embeds the weighted unsigned network to             on signed social network conveys two social relations such as trust
low dimension for visualization by using stochastic neighbor em-         and distrust links in Epinions, and friend and foe links in Slashdot.
bedding. SDNE [34] studies deep networks for network embedding.          In addition to the signed social network, each user is also asso-
DeepWalk [22] introduces the idea of Skip-gram, a word representa-       ciated with a set of attributes. We use X ∈ Rn×m to denote the
tion model in NLP, to learn node representations from random-walk        user-attribute matrix where m is the number of attributes. With the
sequences. Node attributes are also exploited to improve the per-        aforementioned notations and definitions, the problem of signed
formance of unsigned network embedding [7, 46]. For example,             network embedding can be formally stated as follows:
TADW [46] extends DeepWalk to learn network representation by
assuming that each node are associated with rich texts and utilizes      Given a signed social network G with adjacency matrix A ∈ Rn×n
these texts to help network embedding. Gong et. al. [7] performed        and user-attribute matrix X ∈ Rn×m , we aim to learn a low dimen-
joint link prediction and attribute inference using a social-attribute   sional vector representation for each node as
network.
   However, the aforementioned algorithms are designed for un-                                       f (A, X) → U G                           (1)
signed networks and do not take negative links into consideration,
                                                                         where f (·) is the transformation function we want to learn and
while in signed networks there exists both positive and negative
                                                                         U G ∈ Rn×K is the low-dimensional representation of the signed
links. The majority of network embedding algorithms, such as
                                                                         social network with attributes.
spectral analysis, t-SNE, DeepWalk, SDNE and TADW, utilize the
homophily effects or social influence between pairs of linked nodes.
As a result, such pairs are likely to be similar, along with their
                                                                         4   A PRELIMINARY DATA ANALYSIS FOR
vector representations. However, this is not true for signed net-            ATTRIBUTED SIGNED SOCIAL NETWORKS
works due to the existence of negative links; such negative links        Because unsigned network representation learning with attributes
are used to denote distrust or foe relationships between two nodes.      strongly depends on the finding that two linked users are more
Thus, signed network embedding cannot be carried out by simply           likely to share similar attributes than two users without link, it is
                  Table 1: Statistics of the Datasets                        Table 2: Average user attributes similarities in P, N and R
               Dataset               Epinions Slashdot                                          Epinions           Slashdot
               # of Users              27,215    33,407                                      CA COSINE CA COSINE
               # of Positive Links    326,909   477,176                                 P 75.93      0.0650    21.24    0.0332
               # of Negative Links     58,695   158,104                                 N 67.16      0.0540    16.64    0.0289
               # of Reviews/Posts      67,668    94,095                                 R 27.30      0.0168    10.34    0.0162
               # of Attributes         22,367    19,875
               # of Classes              18        22                        We then create the positive link set P, negative link set N and the
                                                                             missing link set R as
natural to explore if similar findings exist on signed social networks
with attributes [7, 15]. Such an understanding lays the groundwork                            P = {(ui , uk )|uk ∈ Pi , i = 1, . . . , n}
for a meaningful framework for signed network embedding with                                  N = {(ui , uk )|uk ∈ Ni , i = 1, . . . , n}       (2)
attributes. In this section, we will first introduce the datasets and                         R = {(ui , uk )|uk ∈ Ri , i = 1, . . . , n}
then perform preliminary data analysis to understand the relations
between signed links and similarities of user attributes.                    We can then calculate the similarity for each pair of users (ui , uk )
                                                                             in P, N and R. In this work, we investigate two ways of calculating
4.1     Datasets                                                             similarity as follows
For the purpose of this study, we collected two datasets from Epin-                • CA: For a pair of users, (ui , uk ), we compute the similarity
ions and Slashdot 1 . Details about the datasets are described below.                sim(ui , uk ) as the number of common attributes by both
   Epinions is a popular product review site. Users in Epinions can                  ui and uk ; and
create both positive (trust) and negative (distrust) links to other                • COSINE: We compute sim(ui , uk ) as the cosine similarity
users, which results in signed network A. They can also write                        between the attributes of ui and attributes of uk
reviews about products. Therefore, reviews written by users are              With these two ways of calculating similarity, we can compute the
used to construct user-attributes matrix X using bag-of-words. We            similarity between each pair of users in P and we use sp ∈ R | P |×1
also collect categories of the products for which they write reviews.        to denote the similarity vector. Similarly, we use sn and sr to denote
For a user who writes reviews for products from multiple categories,         the similarity vectors for N and R, respectively. The mean values
we choose the one with most products she writes reviews to as her            of sp , sn , and sr are shown in Table 2. From the table, we make two
label and there are 18 categories.                                           observations: (i) users are likely to have more similar attributes
   Slashdot is a technology news platform. Users in Slashdot can             with their friends than their foes; and (ii) users are likely to have
create friend (positive) and foe (negative) links to other users, which      more similar attributes with their foes than users without connec-
results in the signed network A. They can also write posts about             tions. The second observation is particularly troubling because it
technologies. The posts written by a user are used to construct              shows that natural extensions of homophily do not apply to signed
the user-attributes matrix A using bag-of-words. We also collect             networks. To statistically verify the observations, we conduct a
information about the groups that they join. These group identifiers         two-sample t-test.
are treated as the class labels and there are 22 classes.
   Some additional preprocessing was performed on these datasets                              Table 3: P-value of t-test results
by filtering users without any positive and negative links, or with                                  Epinions               Slashdot
few non-zero entities in the user-attribute matrix X. A number of                                 CI       COSINE        CI       COSINE
key statistics of these datasets are illustrated in Table 1. It is evident       {sp , sn }   2.31e-31 7.72e-110 3.27e-144 7.85e-148
from the table that (i) positive links are denser than negative links            {sp , sr }   8.51e-151 4.52e-194         0          0
in signed social networks; and (ii) signed networks are very sparse.             {sn , sr }   1.65e-130 6.35e-142 5.972e-198 1.32e-126

4.2     Analysis on Links and Attributes                                       For two vectors {x, y}, the null hypothesis H 0 and the alternative
                                                                             hypothesis H 1 of the two-sample t-test are defined as follows:
Previous studies suggest that users in unsigned social networks are
likely to share similar attributes with their friends, which serves                                 H0 : x ≤ y        H1 : x > y                (3)
as the basis of most network representation with attributes [7, 46].
                                                                             where the null hypothesis indicates that the mean of x is less than
In this subsection, we investigate the attributes similarity of two
                                                                             or equal to that of y. We perform the t-test on {sp , sn }, {sp , sr }
users in signed social networks.
                                                                             and {sn , sr }, respectively, to substantiate the aforementioned ob-
   Let pi , ni and r i denote the number of users with positive, nega-
                                                                             servation. For example, when we perform the t-test on {sp , sn },
tive and no links with ui . We construct three sets for each user ui
                                                                             the null hypothesis is that positively linked users have less com-
with the same size of min(pi , ni , r i ). These sets correspond to (i) a
                                                                             mon attributes than that of negatively linked users; therefore, if
friend circle Pi including randomly selected users who have posi-
                                                                             we reject the null hypothesis, then the assumption that positively
tive links with ui ; (ii) a foe circle Ni containing randomly selected
                                                                             linked have more common attributes than negatively linked users is
users who have negative links with ui ; and (iii) a random circle
                                                                             verified. The null hypothesis for each test is rejected at significance
Ri including randomly selected users who have no links with ui .
                                                                             level α = 0.01 with p-value shown in Table 3, which verifies our
1 http://www.epinions.com/ and https://slashdot.org/                         observations statistically.
     The observation can also be explained from a user’s perspective.           structural balance theory, we have д(ui , uj ) ≤ д(ui , us ) ≤ д(ui , uk ).
Consider a case that ui has more common attributes with u j , which             In terms of user attributes, we have д(ui , uj P) ≤ д(ui , uk P) ≤
could be explained by their common interest in a particular subject             д(ui , us P). In the next section, we will give details of the proposed
such as T echonoloдy. In such a case, it is more likely for ui to be            SNEA.
aware of u j and have interactions such as comments and replies on
u j ’s posts. Thus, ui is more likely to construct positive or negative         5     THE PROPOSED FRAMEWORK
links with users that share attributes with her. If ui has negative             In this section, we give the details of the proposed framework SNEA
link with u j , then they are more likely to hold different opinions            for modeling signed social network with attributes. Since signed
with each other on certain products/things and thus ui ’s attributes            network with attributes has both A and U, we will introduce how
may be less common to u j than ui ’s friends. On the contrary, if               to capture properties of signed network with U and how to capture
ui has few attributes in common with u j , then ui is not likely to             the properties of attributes with UP.
know u j or have interaction with u j . This implies that attributes are
closely related to links and have the potential to help learn better
                                                                                5.1    Modeling Signed Social Network
embedding for link prediction or node clustering.
                                                                                To model signed social networks, one of the most popular and
                                                                                effective methods is to use the low-rank modeling to decompose A
4.3     Extended Structural Balance Theory                                      into low-rank matrices that can be used to reconstruct A [1, 11, 12].
Recently, extended structural balance theory [23] is proposed for               As discussed in the previous section, we use U to denote the latent
comparing the closeness of users in signed social networks with-                user feature matrix for A. Then, the decomposition of A is as
out considering attributes. The essential idea of extended struc-               follows:
tural balance theory is that: for four users ui , u j , uk and us , with                                min kW     (A − UHUT )kF2                       (4)
                                                                                                         U,H
Ai j = 1, Aik = −1 and Ais = 0, i.e., ui has a positive link with
u j , a negative link with uk and no link with us , then д(ui , u j ) ≤         where W ∈ {0, 1}n×n is the indicator matrix with Wi j = 1 if
д(ui , us ) ≤ д(ui , uk ), where д(ui , u j ) means distance between ui         Ai j , 0 and Wi j = 0 otherwise.         is the hadamard operation.
and u j . This can be easily understood from the perspective of the             H ∈ RK ×K is the interaction matrix to capture similarity of user
semantic meanings of the links. For example, if a positive link                 latent features. Ai j is reconstructed as ui HuTj .
means trust and a negative link means distrust, then ui should trust               The extended structural balance theory suggests that a user
u j most, trust uk least and be neutral towards us . Thus, ui prefers           should sit closer to her friends than non-linked users and sit far
to be close to u j and far from uk . If U ∈ Rn×K is the embedding               away from foes. Among various ways to measure closeness, we
learned from A, it should also satisfy the closeness measure, i.e.,             choose the Euclidean distance, which is a popular and effective
д(ui , uj ) ≤ д(ui , us ) ≤ д(ui , uk ), where we use ui to represent the       way to measure distance. With Euclidean distance, д(ui , uj ) =
i-th row of U. However, from the user-attribute perspective, Table              kui − uj k22 . Then, for three users (ui , u j , uk ) with Ai j = 1 and
2 gives д(ui , u j ) ≤ д(ui , uk ) ≤ д(ui , us ), where the first part is the   Aik = 0, we would prefer kui − uj k22 ≤ kui − uk k22 , i.e., the distance
same as extended structural balance theory but the second part                  of embedding satisfies extended structural balance theory. Similarly,
conflicts with extended structural balance theory. If we want the               for Ais = 0 and Aik = −1, we would want kui − us k22 ≤ kui − uk k22 .
embedding to satisfy the manifold structure in terms of attributes and          For each user ui , we randomly select T triplets (ui , u j , uk ) with
also the extended structural balance theory, then using a single em-            Ai j = 1 and Aik = 0, and T triplets triplets (ui , u j , uk ) with Ai j = 0
bedding U will result in conflict. However, if we learn two embedding           and Aik = −1. All these triplets form the set H . We then model
from network and attributes independently, though we can satisfy                extended structural balance theory as follows:
both manifold structure and extended structural balance theory, it
                                                                                                            max(0, kui − uj k22 − kui − uk k22 )
                                                                                                  Õ
raises two problems: (1) We may not capture the inherent connection                       min
                                                                                           U                                                             (5)
between networks and attributes by learning two embeddings and                                (u i ,u j u k )∈H
thus result in non-optimal network representation; and (2) It is more
natural to learn one unified embedding than two embeddings because              For any triplet (ui , u j uk ) ∈ H , we would like to minimize the
a single embedding can be easily used as input to other algorithms for          violation of extended structural balance, which occurs when we
network analysis tasks such as link prediction and node classification.         have kui − uj k22 > kui − uk k22 . In other words, we would like to
Therefore, both singe embedding and two embeddings have there                   minimize kui − uj k22 − kui − uk k22 . There are two cases:
their advantage and disadvantages, which brings challenges for                        • Case 1: if kui −uj k22 ≤ kui −uk k22 , then extended structural
signed network embedding.                                                               balance is satisfied. max(0, kui −uj k22 − kui −uk k22 ) reduces
     This observation suggests that the optimal embedding should                        to 0, which doesn’t penalize w.r.t U and P.
be one unified embedding matrix that captures the inherent rela-                      • Case 2: if kui −uj k22 > kui −uk k22 , then extended structural
tionship between signed network and also satisfies both extended
                                                                                        balance theory is violated. max(0, kui − uj k22 − kui − uk k22 )
social balance theory from networks and manifold constraint from
attributes. Therefore, we propose the dual embedding called SNEA,                       becomes kui − uj k22 − kui − uk k22 , which can be written as
which learns the embedding U ∈ Rn×K capturing the properties                            Tr (Mi jk UUT ), where Mi jk is a sparse n × n matrix with
                                                                                          i jk      i jk    i jk         i jk      i jk    i jk
from signed network and a project matrix P such that UP encodes                         Mi j = Mji = Mkk = −1, Mik = Mki = Mj j = 1 and
the properties from user attributes. With this setting, for extended                    the other entries being 0.
Combining these two cases, we can rewrite Eq.(5) as                            Algorithm 1 Update P
                                Ii jk Mi jk UUT                                Input: Initial feasible P, 0 < µ < 1, 0 < ρ 1 < ρ 2 < 1
                       Õ
               Tr                                                        (6)
                                                

                              (u i ,u j ,u k )∈H                               Output: Updated P
                                                                                 1: Compute G and F
where Ii jk is 1 if kui − uj k22 > kui − uk k22 and 0, otherwise. Putting                                                1
                                                                                 2: Compute Lτ (S(0)) as Lτ (S(0)) = − 2 kFk 2
                                                                                                 0           0
                                                                                                                              F
Eq.(4) and Eq.(6) together, we model signed network with social                  3: set τ = 1
balance theory as follows:                                                       4: for s = 1 to S do

  min kW (A − UHUT )kF2 + αTr                           Ii jk Mi jk UUT                 Compute S(τ ) via Eq.(13)
                                               Õ
                                                                                 5:
                                                                        
                                                                                                     0
  U,H
                                                   (u i ,u j ,u k )∈H            6:     Computer Lτ (S(τ )) via Eq.(16)
                                                             (7)                 7:     if Armijio-Wolfe conditions in Eq.(14) are satisfied then
where α is a scalar to control the contribution of the extended                     break-out
structural balance theory.                                                       8:     τ = µτ
                                                                                 9: end for
5.2        Modeling User Attributes                                             10: Update P as P = S(τ )
                                                                                11: return P
Similarly, matrix factorization is a popular method to learn em-
bedding from attributes. To learn the representations from user
attributes, we project U into the user attribute space with the orthog-
onal projection matrix P, which is popularly used to transform the             where the first term tries to reconstruct the adjacency matrix A; the
latent features from one space to another [8, 25]. With projection             second term captures the local information from signed network,
matrix, we have UP as the user latent features for the user-attributes         and α controls the contribution of this term; the third term mod-
matrix. Then, the decomposition is written as follows:                         els the user-attribute matrix and β controls the importance of this
                                 min        kX − UPVT kF2                (8)   term; the fourth term accounts for the manifold structure based
                             U,V,PT P=I                                        on attributes. The last (regularization) term avoids over-fitting and
where V ∈ Rm×K is the latent feature matrix of attributes. Two                 λ > 0 is a scalar to control the contribution of that term.
users with similar attributes are more likely to be interested in the
same topic or within the same group, which implies that their latent           Discussion One thing worth noting is the orthogonal constraint
representations ui P and uj P should be more similar. This can be              on P. Consider Eq.(11) without the constraint PT P = I and replace
formally written as:                                                           UP with F. We can then optimize the new equation w.r.t U and F
             n Õ n
                                                                               separately. After learning U and F, P can be get as U+ F, where U+
                    1                                                          is the Moore-Penrose pseudoinverse of U. In other words, without
                      Si j kui P − uj Pk22 = min Tr(PT UT LUP)
            Õ
       min                                                         (9)
       U,P
            i=1 j=1
                    2                        U,P                               the constraint, we are actually learning two separate embeddings
                                                                               and U only models A. With the constraint, we avoid the trivial
where S is the similarity matrix with each element defined as Si j =
                                                                               solution as (U+ F)T (U+ F) , I.
           kx −x k 2
exp(− i σ j ) and σ is a scalar to control the similarity. L = D − S
is the Laplacian matrix, and D is a diagonal matrix with the on                6     AN OPTIMIZATION FRAMEWORK
diagonal element Dii = j Si j . Thus, if two nodes are more similar,
                             Í
                                                                               The objective function in Eq.(11) is not convex, so we cannot update
(i.e., Si j is large), we penalize more with Si j kui P − uj Pk22 and thus     all the variables jointly. To optimize the objective function, we use
ui P and uj P are closer. With Eq.(8) and Eq.(9), we model user-               alternating optimization, which is a popular method used in matrix
attributes with graph regularization as follows:                               factorization [12]. Specifically, we optimize one set of variables in
                       min    kX − UPVT kF2 + γ Tr(PT UT LUP)           (10)   the factors, by fixing other variables. Next, we give the details to
                 U,V,PT P=I                                                    optimize the objective function followed by complexity analysis.
where γ is a scalar to control the contribution of graph regularizer.
                                                                               6.1    Update Rules
5.3        Proposed Model – SNEA                                               For simplicity, we denote the objective function in Eq.(11) as L.
We have introduced our approaches to model network and at-
tributes separately. With these two components, we propose the                    6.1.1 Update Rule for P. It is generally difficult to optimize w.r.t
signed network embedding framework SNEA, which exploits both                   P due to the orthogonal constraint. In this work, we use a gradient
network and attributes for network representation. The proposed                descent optimization procedure with curvilinear search [44] to
SNEA framework solves the following optimization problem:                      solve it. In each iteration of the gradient descent procedure, given
                                                                               the current feasible point P, we define G as
  min kW (A − UHUT )kF2 + αTr                   Ii jk Mi jk UUT
                                        Õ                       
U,P,V,H
                                                   (u i ,u j ,u k )∈H                        ∂L
                                                                                        G=      = 2(βUT UPVT V − βUT XV + γ UT LUP)              (12)
      +β kX − UPVT kF2 + γ Tr(PT UT LUP) + λ(kUkF2 + kVkF2 + kHkF2 )                         ∂P
           T
   s.t .      P P=I                                                            Let F = GPT − PGT . It is easy to verify that FT = −F and thus
                                                                        (11)   F is skew-symmetric. The next new point can be searched as a
curvilinear function of a step size variable τ , such that                  Algorithm 2 Signed Network Embedding with Attributes
                                τ          τ                                Input: A ∈ Rn×n , X ∈ Rn×m , α, β, γ , λ, K
                   S(τ ) = (I + F)−1 (I − F)Q                       (13)
                                2          2                                Output: U ∈ Rn×K , P ∈ RK ×K
Given that F is skew-symmetric, it is easy to prove that S(τ )T S(τ ) =      1: Initialize U, P, H and V
I using the Cayley transformation [14]. Thus we can stay in the              2: repeat
feasible region along the curve defined by τ . We thus apply a               3:     calculate M using Eq.(22)
similar strategy as the standard back-tracking line search to find a         4:     update U as U ← U − ϵ ∂∂U L using Eq.(17)

proper step size τ using curvilinear search, while guaranteeing the          5:     update P as with Algorithm 1
iterations to converge to a stationary point. We determine a proper          6:     update H as H ← H − ϵ ∂∂H L using using Eq.(20)

step size τ as one satisfying the following Armijo-Wolfe conditions:         7:
                                                                                                           ∂ L
                                                                                    update V as V ← V − ϵ ∂V using Eq.(21)
  L(S(τ )) ≤ L(S(0)) + ρ 1τ Lτ0 (S(0)), Lτ0 (S(τ )) ≥ ρ 2 Lτ0 (S(0)) (14)    8: until Convergence
        0                                    0                               9: return U, P, Q, H
Here Lτ (S(τ )) is the derivative of Lτ (S(τ )) w.r.t τ ,
                                         τ       P + S(τ ) 
          Lτ (S(τ )) = −Tr R(τ )T (I + F)−1 F
             0
                            
                                                                    (15)
                                         2             2                    1 without actually creating Mi jk . Thus, we don’t need to create
             ∂ L(S(τ ))
where R(τ ) = ∂S(τ ) . Obviously, S(0) = P and thus                         |H | matrices.
                                                                                The convergence of the algorithm using alternating optimization
                                ∂L(P)                                       is guaranteed [3]. This is because each time we use gradient descent
                         R(0) =         =G                        (16)
                                  ∂P                                        to update the parameters, we monotonically reduce the value of the
                                                                            objective function. Since the value of objective function in Eq.(11)
                                               = − 12 kFkF2 . We sum-
                                            
Therefore, Lτ (S(0)) = −Tr R(0)T F 2
              0                       Q+S(0)
                                                                            is non-negative, the algorithm will converge and we will arrive at
marize the update rule for P in Algorithm 1, where S is the maximal         a local optimum.
iterations for the loop.
   6.1.2 Update Rules for U, H and V. The gradient of L w.r.t               6.3    Time Complexity Analysis
U, P, H and V are given as follows                                          The algorithm is composed of two parts, i.e., initialization and
   1 ∂L                                                                     updating parameters. The most time-consuming part is the updating
        = −(W W A)UHT + βUPVT VPT − βXVPT                           (17)    part. Thus, our analysis will focus on updating parameters. First,
   2 ∂U
                                                                            the cost of calculating M using Eq.(22) is O(K |H |). Considering
        − (W W A)T UH + (W W UHUT )UHT                              (18)
                                                                            the fact that W and A are very sparse, it is evident that W W A
            + (W   W    UHUT )T UH + αMU + γ LUPPT + λU             (19)    and W W UHUT are also very sparse. Thus, in Eq.(17), the
   1 ∂L                                                                     computational cost of (W W A)UHT is about O(nK 2 ) and the
        = −UT (W        W      A)U + λH + UT (W       W     UHUT )U         computational cost of (W W UHUT )UHT is O(nK 2 +n 2 K). The
   2 ∂H
                                                                    (20)    costs of calculating UPVT VP and XVPT are both O(nmK + nK 2 )
  1 ∂L                                                                      and cost of MU is O(|H |K) as M is sparse. Therefore, the cost
       = βVPT UT UP − βXT UP + λV                                   (21)    of calculating ∂∂UL is O(nK 2 + n 2 K + nmK + K |H |) and updating
  2 ∂V
where M is defined as follows:                                              U is O(nK). With a similar analysis procedure, we can get the
                                                                            computational cost of updating the other parameters. We omit
                                Ii jk Mi jk
                             Õ
                   M=                                               (22)    the details here and directly give the cost. The cost of updating P
                            (u i ,u j ,u k )∈H                              Algorithm 1 is O(nK 2 + n 2 K + nmK + SK 3 ). The cost of calculating
                                                                             ∂ L is O(nK 2 + n 2 K). And the costs of calculating ∂ L is O(nmK +
Then the parameters can be updated θ ← θ − ϵ ∂∂θL , where θ =                ∂H                                                   ∂Q
{U, H, Q} and ϵ is the learning rate                                        mK 2 ). Thus, the cost of the algorithm in one iteration is O(nmK +
                                                                            nK 2 + n 2 K + mK 2 + SK 3 + K |H |). Since K is usually much smaller
6.2    Learning Algorithm of SNEA                                           than n and m and we set S to be 10 in practice, the cost nK 2 , mK 2 and
With the update rules of U, P, H and V given above, the optimization        SK 3 can be ignored compared to nmK and n2 K. Since H is formed
algorithm for SNEA is shown in Algorithm 2. Next we briefly review          by 2T triplets for each user, we have |H | = O(nT ). Therefore, the
Algorithm 2. In Line 1, we first randomly initialize the parameters         total cost of the algorithm is O(t(nmK + n 2 K + nT K)), where t is
U, P, H and V. With U, we can calculate M. After initialization,            the number of iterations.
from Line 2 to Line 9, we update these parameters sequentially
until convergence. Finally, the embedding of the network is given           7     EXPERIMENTAL RESULTS
as U, which can facilitate other network mining tasks such as signed        In this section, we conduct experiments to evaluate the effective-
link prediction and node clustering.                                        ness of the proposed framework SNEA and factors that could affect
   The calculation of M using Eq.(22) involves |H | sparse matrices.        the performance of SNEA. To measure the quality of the embedding
To save memory, we can first initialize M to be a all zero matrix. We       learned by SNEA, following the common way [2, 47], we use the
then update M as : if (ui , u j , uk ) ∈ H gives Ii jk = 1, we decrease     embedding for two popular tasks of mining signed social networks,
values of Mi j , Mji , Mkk by 1 and increase values Mik , Mki , Mj j by     i.e., signed link perdition and node clustering, with comparisons
to state-of-the-art baseline methods. Further experiments are con-                 • CMF: Collective matrix factorization [26] is a matrix fac-
ducted to study parameter sensitivity.                                               torization model that jointly utilizes signed network A and
                                                                                     user attributes X as
7.1    Signed Link Prediction on Signed Network                                           min kW         (A − UV1T )kF2 + α kX − UVT2 kF2
                                                                                         U,V1,V2
                                                                                                                                                 (23)
In this subsection, we check whether the learned embedding can
                                                                                                   + λ(kUkF2 + kV1 kF2 + kV2 kF2 )
improve the performance of signed link prediction. We begin by
introducing the experimental settings.                                                 Signed links are then predicted as UVT1 . It is a naive way
                                                                                       of modeling both links and attributes without considering
   7.1.1 Experimental Settings. Let T = {< ui , u j > |Ai j = 1} be                    extend social balance theory and attributes similarity.
the set of users having positive links and D = {< ui , u j > |Ai j =                • SESN: Spectral embedding of signed network [47] is a nor-
−1} be the set of users having negative links. For both Epinions and                   malized spectral analysis method for signed graphs. It
Slashdot datasets, we randomly select 20% positive and negative                        defines signed Laplacian via Rayleigh quotients and com-
links from T and D, respectively, which are used as testing set.                       putes the top-k eigenvectors as node representation.
We then remove the 20% selected links from T and D. For the                         • ELLR: ELLR [27] is the state-of-the-art ranking based signed
remaining links in T and D, we choose x% positive and negative                         network embedding algorithm which put more emphasis
links from T and D, respectively, as training set. We vary x as                        on ranking positive links on the top and negative links at
{20, 40, 60, 80, 100}. The purpose of varying the values of x is to                    the bottom of a ranking list.
investigate the performance of the proposed framework on the two                    • SiNE: SiNE [36] is a deep network based method for signed
datasets with different statistics. In real-world signed social net-                   network embedding. It tries to push a user closer to his
works such as Epinions and Slashdot, positive links are often much                     friends but faraway from his foes. No attributes are used.
denser than negative links; hence positive and negative links are                   • ESiNE: Enhanced SiNE which aggregates the embedding
imbalanced in both training and testing sets. Therefore, following                     of SiNE and the embedding of attributes using matrix fac-
the common way to evaluate the signed link prediction problem, we                      torization. This method also uses both link and attributes.
use AUC and F1 instead of accuracy to assess the performance [16].               Since distrust (negative link) is not the negation of trust (positive
AUC [6] measures the probability that the classifier will rank a             link), the trust/distrust prediction problem cannot be successfully
randomly chosen positive instance higher than a randomly chosen              carried out by directly applying trust predictors [30]. Therefore we
negative one; a higher AUC would indicate a better predictive per-           do not compare SNEA with traditional trust predictors such as [13,
formance. F1 score accounts for the trade-off between precision              29]. Note that we use cross-validation to determine parameters for
and recall; and higher F1 score indicates higher predictive power.           all baseline methods. For SNEA, α, γ and λ are set to 0.1, β is set to
                                                                             1, K = 20 and T = 30. More details about parameter selection will
   7.1.2 Performance Comparison of Signed Link Prediction. We                be discussed in the following subsections. We use U to reconstruct
compare the proposed framework SNEA with the state-of-the-art                A as Ã = UHUT for signed link prediction. Each experiments are
signed link prediction methods, where MF, SESN and ELLR only                 conducted 5 times and the average performance are reported in
use signed network while FExtra, CMF and ESiNE use both signed               Figure 1. From these figures, we make the following observations:
network and attribute information. The details of the compared
methods are:                                                                        • In general, with the increase of the training data, the per-
                                                                                       formance of all methods increases.
      • FExtra: FExtra is a variant of the feature based method                     • Among MF, SESN, ELLR and SiNE, generally, SiNE out-
        proposed in [16]. The original method in [16] extracts 23                      performs the other three. This is because SiNE is a deep
        features for each pair of nodes from signed network for                        network based method which tries to learning embeddings
        link prediction. Specifically, for each pair of users (ui , u j ),             such that a user is more closer to his friends than his foes,
        it extracts two types of features, i.e., degree based and triad                which is more efficient. However, SNEA outperforms SiNE,
        based features. Degree based features contain the degree                       which is because (i) SNEA uses extended social balance
        information such as the number of incoming positive and                        theory to guide the representation learning process; and
        negative links of ui , the number of outgoing positive and                     (2) SNEA incorporate attributes, which provides comple-
        negative links of u j and so on; while triad based features                    mentary information.
        include structure information of triads that contains ui and                • By introducing attributes, ESiNE outperforms SiNE, which
        u j , such as number of common users. For each pair of users,                  indicates attributes are useful for signed link prediction.
        FExtra uses both the 23 features and their attributes to train              • The proposed framework SNEA always obtain the best
        a logistic regression classifier for singed link prediction.                   performance. Though, FExtra, CMF ESiNE also exploit
        Note that this feature based method uses both signed links                     signed network and attributes for signed link prediction,
        and node attributes.                                                           SNEA outperforms them, which suggests that SNEA is
      • MF: Matrix factorization [12] based method which fac-                          more effective in modeling signed network and attributes
        torizes the adjacency matrix A into two low rank latent                        simultaneously. This is because SNEA takes extended social
        matrices and predicts the links by the matrix reconstructed                    balance theory and attribute similarity into consideration
        by the two low rank matrices. This can be seen as variant                      and thus can learn better embedding; while ESiNE simply
        of SNEA without considering attributes.                                        aggregates two embeddings.
               0.96                                                             0.97


                                                                                0.96
               0.94

                                                                                0.95
               0.92

                                                                                0.94
                0.9

         AUC                                                               F1   0.93
               0.88
                                                                FExtra          0.92                                               FExtra
                                                                MF                                                                 MF
               0.86                                             CMF                                                                CMF
                                                                                0.91                                               SESN
                                                                SESN
                                                                ELLR                                                               ELLR
               0.84                                             SiNE             0.9                                               SiNE
                                                                ESiNE                                                              ESiNE
                                                                SNEA                                                               SNEA
               0.82                                                             0.89
                  20%      40%             60%            80%       100%           20%         40%            60%            80%       100%
                                    x% of Training Data                                                x% of Training Data

                                 (a) AUC on Epinions                                                 (b) F1 on Epinions
               0.94                                                             0.93


                                                                                0.92
               0.92


                                                                                0.91
                0.9

                                                                                 0.9

         AUC   0.88                                                        F1
                                                                                0.89
                                                                FExtra                                                             FExtra
               0.86                                             MF                                                                 MF
                                                                CMF             0.88                                               CMF
                                                                SESN                                                               SESN
               0.84                                             ELLR                                                               ELLR
                                                                SiNE            0.87                                               SiNE
                                                                ESiNE                                                              ESiNE
                                                                SNEA                                                               SNEA
               0.82                                                             0.86
                  20%      40%             60%            80%       100%           20%         40%            60%            80%       100%
                                    x% of Training Data                                                x% of Training Data

                                 (c) AUC on Slashdot                                                 (d) F1 on Slashdot
        Figure 1: Performance comparison of signed link prediction on Epinions and Slashdot in terms of AUC and F1
   We conduct t-test on all performance comparisons and it is evi-              both datasets are imbalanced. Some classes have more than 2,000
dent from t-test that all improvements are significant. In summary,             users, while some only has around 500 hundred users.
SNEA obtains significant performance improvement in signed link
prediction by leveraging both signed network and user attributes
and considering the attributes similarity and semantic meanings of                 7.2.2 Performance Comparison of Node Clustering. We compare
signed links.                                                                   the proposed framework with state-of-the-art signed network em-
                                                                                bedding and feature based methods. The compared algorithms are
7.2    Node Clustering on Signed Social Network                                 FExtra, MF, CMF, SNSE, ELLR, SiNE and ESiNE, which are intro-
                                                                                duced in section 7.1.2. Note that FExtra, CMF and ESiNE utilizes
In this subsection, we further check whether the learned signed
                                                                                both the singed network and attributes. For FExtra, we want the
network embedding with attributes can improve the performance
                                                                                representation of each node instead of features for each pair for
of node clustering. We first introduce experimental settings.
                                                                                clustering. Thus, we only extract degree based features from signed
   7.2.1 Experimental Settings. In this experiment, we also use                 network as triad based features are designed for pairs of nodes.
Epinions and Slashdot for node clustering. Specifically, for both               These manually extracted features and attributes are used as node
datasets, we first learn the signed network embedding. With the                 representation for node clustering. For the other methods, we first
learned embedding, K-means is applied on the embedding to cluster               learn embedding and then use embedding for node clustering.
the nodes into C clusters. Following the common way to measure                     There are some parameters to be set. For MF, CMF and SNEA, we
the quality of clusters [38, 40], two widely used evaluation metrics,           need to set the latent dimension K. We tune K for these methods by
accuracy (ACC) and normalized mutual information (NMI), are                     a “grid-search” strategy from {20, 40, 60, 100}. For SiNE, we use the
employed. The larger ACC and NMI are, the better performance is.                default setting as in [36]. For SNEA, we empirically set α, γ and λ to
Since K-means depends on initialization, following previous work,               0.1, β to 1 and T = 30. More details about parameter selection will
we repeat the experiments 20 times and the average results with                 be discussed in the following subsections. The average experimental
standard deviation are reported. As shown in Table 1, Epinions has              results of different methods on the datasets are summarized in Table
18 classes and Slashdot has 22 classes. One thing to note is that               4 and 5. From these two tables, we make the following observations:
             Table 4: Node clustering results (ACC±std) of different algorithms on different Epinions and Slashdot.
 Dataset           MF           CMF          FExtra         SESN          ELLR          SiNE         ESiNE          SNEA
 Epinions     0.1835±0.0149 0.2129±0.0187 0.1868±0.0159 0.1829±0.0156 0.2059±0.0164 0.2097±0.0274 0.2189±0.0241 0.2453±0.0148
 Slashdot     0.2274±0.0197 0.2585±0.0205 0.2398±0.0176 0.2301±0.0216 0.2503±0.0201 0.2598±0.0225 0.2686±0.0196 0.2869±0.0197

             Table 5: Node clustering results (NMI±std) of different algorithms on different Epinions and Slashdot.
 Dataset           MF           CMF          FExtra         SESN          ELLR          SiNE         ESiNE          SNEA
 Epinions     0.1329±0.0058 0.1627±0.0079 0.1359±0.0048 0.1314±0.0052 0.1549±0.0062 0.1628±0.0074 0.1701±0.0068 0.1825±0.0049
 Slashdot     0.1576±0.0087 0.1845±0.0089 0.1692±0.0078 0.1602±0.0093 0.1739±0.0084 0.1796±0.0092 0.1879±0.0082 0.2065±0.0088


      • For both datasets, CMF outperforms MF, and similarly,             vary the values of β as {0.001, 0.01, 0.1, 1, 10} and the values of γ
        ESiNE outperforms SiNE. CMF and ESiNE exploit both                as {0.001, 0.01, 0.1, 1, 10}. The results for signed link prediction and
        signed network and attributes while their variants MF and         node clustering are shown in Figure 2 and 3, respectively. From
        SiNE only utilize signed network. This suggests that at-          these figures, we make the following observations: (1) Generally, as
        tributes are helpful for node clustering. There are two rea-      the increase of K, the performance first increase and then decrease
        sons: (1) The signed social network is very sparse, which         after K reaches certain value. The same holds for α, β and γ , which
        makes it difficult for node clustering; and (2) Users at-         is because when β and γ are small, the contribution of attributes
        tributes provides complementary information such as users         becomes small; and (2) The performance is generally better and
        interests and properties, which are not available in signed       more stable when the value of K is in [20, 40], and the value of α is
        social networks and can mitigate network sparsity prob-           in [0.01, 10]. Similarly, a value of α (β) within [0.01, 10] gives better
        lem.                                                              performance. This observation eases the parameter selection.
      • For the algorithms that don’t consider node attributes, i.e.,
        SESN, MF, ELLR and SiNE, SiNE outperforms the other
        three. However, SNEA outperforms SiNE, which implies
                                                                          0.96
        that by considering attributes, we can learn better node          0.94
                                                                                                                                   0.96

                                                                                                                                   0.94
        representation for clustering.                                    0.92                                                     0.92                                                   10
      • The proposed framework SNEA outperforms all the base-              0.9
                                                                                                                             100
                                                                                                                                    0.9                                            1
                                                                          1e−3                                          40
        line methods. We also conduct t-test on all performance                  1e−2
                                                                                        1e−1                   20
                                                                                                                                     10
                                                                                                                                          1
                                                                                                                                                                               1e−1
                                                                                                                                              1e−1
        comparisons and it is evident that improvements are sig-                                   1
                                                                                                       10 10        K                                    1e−2
                                                                                                                                                                       1e−3
                                                                                                                                                                           1e−2
                                                                                                                                                                               γ
                                                                                               α                                                     β          1e−3
        nificant, which demonstrates the quality of the learned
                                                                          (a) AUC on Epinions with β = 1 and γ = (b) AUC on Epinions with K = 20 and
        network embedding. In particular, SNEA, CMF, ESiNE and            0.1                                    α = 0.1
        FExtra use network and attributes for constructing node
        representation. SNEA achieves better performance than
        the other three. This is because FExtra extracts features sep-
        arately from network and attributes; CMF doesn’t consider         0.94                                                     0.94
                                                                          0.92
        extended structural balance theory and attributes manifold         0.9
                                                                                                                                   0.92

                                                                                                                                    0.9
        structure; ESiNE simply aggregates network embedding              0.88
                                                                                                                             100                                                          10
                                                                                                                                   0.88                                               1
        from SiNE and features from MF on attributes; while SNEA          1e−3
                                                                                 1e−2
                                                                                                                        40
                                                                                                                                     10
                                                                                                                                          1                                    1e−1
                                                                                        1e−1                   20
        leverages both network and attributes into a unified frame-                                1
                                                                                                                    K
                                                                                                                                              1e−1
                                                                                                                                                         1e−2
                                                                                                                                                                           1e−2
                                                                                                       10 10                                                           1e−3    γ
        work by considering extended structural balance theory                                 α                                                     β          1e−3


        and attributes manifold structure.                                (c) AUC on Slashdot with β = 1 and γ = (d) AUC on Slashdot with K = 20 and
                                                                          0.1                                    α = 0.1
                                                                          Figure 2: Parameter Sensitivity for SNEA on Link Prediction
7.3    Parameter Analysis for SNEA
The proposed framework has four importance parameters, i.e., α            8      CONCLUSION
controlling the contribution of extended social balance theory; β         In this paper, we investigate the problem of signed social network
and γ controls the contribution of attributes; and K controls the         embedding with attributes. We first study the attributes similarity of
number of dimension. In this section, we investigate the impact           users with positive, negative and no links, which lays the foundation
of these parameters on the link prediction and node clustering            of the proposed framework SNEA. We then leverage both signed
performances of SNEA. We use the same experimental settings as            social network and user attributes into a unified framework SNEA
previous section. For signed link prediction, we only show the            by incorporating the extended structure balance theory and the
results for x = 100 in terms of AUC as we have similar observations       relationship between user links and user attributes. Experiments
for x = {20, 40, 60, 80} and F1 metric. Similarly, for node clustering,   on real-world datasets demonstrate that the learned embedding by
we only show results in terms of ACC. We first fix β to be 1, γ to be     leveraging signed network and user attributes outperforms signed
0.1 and vary the values of α as {0.001, 0.01, 0.1, 1, 10}, the values     network embedding without attributes in signed network mining
of K as {10, 20, 40, 100}. We then fix α to be 0.1, K to be 20 and        tasks such as signed link prediction and node clustering.
                                                                                                                                       [12] Cho-Jui Hsieh, Kai-Yang Chiang, and Inderjit S Dhillon. 2012. Low rank modeling
                                                                                                                                            of signed networks. In SIGKDD. ACM, 507–515.
                                                                     0.24
 0.24
                                                                                                                                       [13] Jin Huang, Feiping Nie, Heng Huang, and Yi-Cheng Tu. 2012. Trust prediction
 0.23
                                                                     0.23                                                                   via aggregating heterogeneous social networks. In CIKM. ACM.
                                                                     0.22                                                              [14] Alan Jennings and John J McKeown. 1992. Matrix computation. John Wiley &
 0.22                                                          100
                                                                     0.21                                                                   Sons Inc.
                                                                                                                                  10
 0.21
 1e−3
                                                          40           10
                                                                                                                              1
                                                                                                                                       [15] Timothy La Fond and Jennifer Neville. 2010. Randomization tests for distinguish-
                                                                            1
        1e−2
                1e−1
                                              20                                1e−1                               1e−1                     ing social influence and homophily effects. In WWW. ACM, 601–610.
                               1                      K                                 1e−2           1e−2                            [16] Jure Leskovec, Daniel Huttenlocher, and Jon Kleinberg. 2010. Predicting positive
                                     10 10
                           α                                                           β     1e−3 1e−3     γ                                and negative links in online social networks. In WWW. ACM, 641–650.
(a) ACC on Epinions with β = 1 and γ = (b) ACC on Epinions with K = 20 and                                                             [17] Jure Leskovec, Daniel Huttenlocher, and Jon Kleinberg. 2010. Signed networks
0.1                                    α = 0.1                                                                                              in social media. In SIGCHI. ACM, 1361–1370.
                                                                                                                                       [18] David Liben-Nowell and Jon Kleinberg. 2007. The link-prediction problem for
                                                                                                                                            social networks. Journal of the American society for information science and
                                                                                                                                            technology 58, 7 (2007), 1019–1031.
0.29
                                                                                                                                       [19] Silviu Maniu, Bogdan Cautis, and Talel Abdessalem. 2011. Building a signed
0.28
                                                                     0.29                                                                   network from interactions in Wikipedia. In Databases and Social Networks.
                                                                     0.28                                                              [20] Mingdong Ou, Peng Cui, Jian Pei, Ziwei Zhang, and Wenwu Zhu. 2016. SIGKDD.
0.27
                                                                     0.27                                                                   1105–1114.
0.26
                                                               100   0.26                                                         10   [21] Symeon Papadopoulos, Yiannis Kompatsiaris, Athena Vakali, and Ploutarchos
0.25
1e−3                                                      40
                                                                       10                                                     1             Spyridonos. 2012. Community detection in social media. Data Mining and
                                                                            1
        1e−2
               1e−1                          20                                  1e−1
                                                                                                                       1e−1                 Knowledge Discovery 24, 3 (2012), 515–554.
                           1
                                   10 10          K                                         1e−2               1e−2                    [22] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. Deepwalk: Online
                                                                                                                   γ
                       α                                                                β          1e−3 1e−3                                learning of social representations. In SIGKDD. ACM, 701–710.
(c) ACC for Slashdot with β = 1 and γ = (d) ACC on Slashdot with K = 20 and α =                                                        [23] Yi Qian and Sibel Adali. 2013. Extended structural balance theory for modeling
0.1                                     0.1                                                                                                 trust in social networks. In PST. IEEE, 283–290.
                                                                                                                                       [24] Sam T Roweis and Lawrence K Saul. 2000. Nonlinear dimensionality reduction
Figure 3: Parameter Sensitivity for SNEA on Node Cluster-                                                                                   by locally linear embedding. Science 290, 5500 (2000), 2323–2326.
ing                                                                                                                                    [25] Xiaoxiao Shi, Qi Liu, Wei Fan, S Yu Philip, and Ruixin Zhu. 2010. Transfer
                                                                                                                                            learning on heterogenous feature spaces via spectral transformation. In ICDM.
                                                                                                                                            IEEE, 1049–1054.
   There are several interesting directions need further investiga-                                                                    [26] Ajit P Singh and Geoffrey J Gordon. 2008. Relational learning via collective
tion. First, in this work, we only consider signed link prediction and                                                                      matrix factorization. In SIGKDD. ACM, 650–658.
node clustering. One future work is to use the learned embedding                                                                       [27] Dongjin Song, David A Meyer, and Dacheng Tao. 2015. Efficient latent link
                                                                                                                                            recommendation in signed networks. In SIGKDD. ACM, 1105–1114.
for other network mining tasks such as signed network visualiza-                                                                       [28] Jiliang Tang, Yi Chang, Charu Aggarwal, and Huan Liu. 2016. A Survey of Signed
tion. Second, user links and user attributes have close relationship                                                                        Network Mining in Social Media. ACM Comput. Surv. 49, 3 (Aug. 2016).
                                                                                                                                       [29] Jiliang Tang, Huiji Gao, Xia Hu, and Huan Liu. 2013. Exploiting homophily effect
and user attributes are helpful for link prediction. Thus, another di-                                                                      for trust prediction. In WSDM. ACM, 53–62.
rection is to investigate if signed links are useful for user attributes                                                               [30] Jiliang Tang, Xia Hu, and Huan Liu. 2014. Is distrust the negation of trust?: the
prediction. In addition, visual features have been demonstrated to                                                                          value of distrust in social media. In Proceedings of the 25th ACM conference on
                                                                                                                                            Hypertext and social media. ACM, 148–157.
be very effective for data mining [41–43]. Therefore, we also want                                                                     [31] Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei.
to exploit visual features for signed network embedding.                                                                                    2015. LINE: Large-scale Information Network Embedding. In WWW. 1067–1077.
                                                                                                                                       [32] Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using
                                                                                                                                            t-SNE. Journal of Machine Learning Research 9, 2579-2605 (2008), 85.
9       ACKNOWLEDGEMENTS                                                                                                               [33] Patricia Victor, Chris Cornelis, Martine De Cock, and Ankur Teredesai. 2009.
This material is based upon work supported by, or in part by, the                                                                           Trust-and distrust-based recommendations for controversial reviews. (2009).
                                                                                                                                       [34] Daixin Wang, Peng Cui, and Wenwu Zhu. 2016. Structural Deep Network
National Science Foundation (NSF) under the grant #1614576 and                                                                              Embedding. In SIGKDD. 1225–1234.
Office of Naval Research (ONR) under the grant N00014-16-1-2257.                                                                       [35] Jing Wang, Jie Shen, Ping Li, and Huan Xu. 2017. Online Matrix Completion for
                                                                                                                                            Signed Link Prediction. In WSDM.
                                                                                                                                       [36] Suhang Wang, Jiliang Tang, Charu Aggarwal, Yi Chang, and Huan Liu. 2017.
REFERENCES                                                                                                                                  Signed Network Embedding in Social Media. In SDM.
 [1] Ghazaleh Beigi, Jiliang Tang, Suhang Wang, and Huan Liu. 2016. Exploiting                                                         [37] Suhang Wang, Jiliang Tang, Charu Aggarwal, and Huan Liu. 2016. Linked
     emotional information for trust/distrust prediction. In SDM.                                                                           document embedding for classification. In CIKM. ACM.
 [2] Mikhail Belkin and Partha Niyogi. 2001. Laplacian eigenmaps and spectral                                                          [38] Suhang Wang, Jiliang Tang, and Huan Liu. 2015. Embedded Unsupervised
     techniques for embedding and clustering.. In NIPS, Vol. 14. 585–591.                                                                   Feature Selection.. In AAAI. Citeseer, 470–476.
 [3] James C Bezdek and Richard J Hathaway. 2003. Convergence of alternating                                                           [39] Suhang Wang, Jiliang Tang, Fred Morstatter, and Huan Liu. 2016. Paired restricted
     optimization. Neural, Parallel & Scientific Computations (2003).                                                                       boltzmann machine for linked data. In CIKM. ACM.
 [4] Smriti Bhagat, Graham Cormode, and S Muthukrishnan. 2011. Node classification                                                     [40] Suhang Wang, Yilin Wang, Jiliang Tang, Charu Aggarwal, Suhas Ranganath,
     in social networks. In Social network data analytics. Springer, 115–148.                                                               and Huan Liu. 2017. Exploiting hierarchical structures for unsupervised feature
 [5] Yuxiao Dong, Jing Zhang, Jie Tang, Nitesh V Chawla, and Bai Wang. 2015.                                                                selection. In SDM. SIAM.
     CoupledLP: Link Prediction in Coupled Networks. In SIGKDD. ACM, 199–208.                                                          [41] Suhang Wang, Yilin Wang, Jiliang Tang, Kai Shu, Suhas Ranganath, and Huan Liu.
 [6] Tom Fawcett. 2006. An introduction to ROC analysis. Pattern recognition letters                                                        2017. What your images reveal: Exploiting visual contents for point-of-interest
     27, 8 (2006), 861–874.                                                                                                                 recommendation. In WWW. 391–400.
 [7] Neil Zhenqiang Gong, Ameet Talwalkar, Lester Mackey, Ling Huang, Eui                                                              [42] Yilin Wang, Suhang Wang, Jiliang Tang, Huan Liu, and Baoxin Li. 2015. Unsu-
     Chul Richard Shin, Emil Stefanov, Elaine Runting Shi, and Dawn Song. 2014.                                                             pervised Sentiment Analysis for Social Media Images.. In IJCAI. 2378–2379.
     Joint link prediction and attribute inference using a social-attribute network.                                                   [43] Yilin Wang, Suhang Wang, Jiliang Tang, Huan Liu, and Baoxin Li. 2016. Ppp:
     ACM Transactions on Intelligent Systems and Technology (TIST) 5, 2 (2014), 27.                                                         Joint pointwise and pairwise image label prediction. In CVPR.
 [8] Yuhong Guo and Min Xiao. 2012. Cross language text classification via subspace                                                    [44] Zaiwen Wen and Wotao Yin. 2013. A feasible method for optimization with
     co-regularized multi-view learning. In ICML.                                                                                           orthogonality constraints. Mathematical Programming 142, 1-2 (2013), 397–434.
 [9] Ahmed Hassan, Amjad Abu-Jbara, and Dragomir Radev. 2012. Extracting signed                                                        [45] Bo Yang, Xuehua Zhao, and Xueyan Liu. 2015. Bayesian Approach to Modeling
     social networks from text. In ACL Workshop on Graph-based Methods for NLP.                                                             and Detecting Communities in Signed Network.. In AAAI. 1952–1958.
[10] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng                                                          [46] Cheng Yang, Zhiyuan Liu, Deli Zhao, Maosong Sun, and Edward Y Chang. 2015.
     Chua. 2017. Neural collaborative filtering. In WWW.                                                                                    Network representation learning with rich text information. In IJCAI. 2111–2117.
[11] Xiangnan He, Hanwang Zhang, Min-Yen Kan, and Tat-Seng Chua. 2016. Fast                                                            [47] Q Zheng and DB Skillicorn. 2015. Spectral embedding of signed networks. In
     matrix factorization for online recommendation with implicit feedback. In SIGIR.                                                       SDM. SIAM, 55–63.

