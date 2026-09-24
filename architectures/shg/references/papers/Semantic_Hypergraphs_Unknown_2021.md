# Semantic Hypergraphs Unknown 2021

> Source: `Semantic_Hypergraphs_Unknown_2021.pdf`

---

                                                                                 Semantic Hypergraphs

                                                                              Telmo Menezes* 1 and Camille Roth†1,2
                                               1 Computational Social Science Team, Centre Marc Bloch Berlin (CNRS/HU), Friedrichstr. 191, 10117 Berlin, Germany
                                                  2 CAMS (Centre Analyse et Mathématique Sociales, UMR 8557 CNRS/EHESS), 54 Bd Raspail, 75007 Paris, France




                                         Abstract                                                          1 Introduction
                                                                                                           Natural language processing (NLP) approaches generally
                                         Approaches to Natural language processing (NLP) may
                                                                                                           belong to either one of two main strands, which also of-
                                         be classified along a double dichotomy open/opaque –
                                                                                                           ten appear to be mutually exclusive. On the one hand




arXiv:1908.10784v2 [cs.IR] 18 Feb 2021
                                         strict/adaptive. The former axis relates to the possibility
                                                                                                           we essentially have symbolic methods and models which
                                         of inspecting the underlying processing rules, the latter
                                                                                                           are open, in the sense that their internal mechanisms as
                                         to the use of fixed or adaptive rules. We argue that many
                                                                                                           well as their conclusions are easy to inspect and under-
                                         techniques fall into either the open-strict or opaque-
                                                                                                           stand, but which deal with linguistic patterns in a rela-
                                         adaptive categories. Our contribution takes steps in the
                                                                                                           tively strict manner. On the other hand, we have adap-
                                         open-adaptive direction, which we suggest is likely to
                                                                                                           tive models based on machine learning (ML) which are
                                         provide key instruments for interdisciplinary research.
                                                                                                           usually opaque to inspection and too complex for their
                                         The central idea of our approach is the Semantic Hyper-
                                                                                                           reasoning to be intelligible, but which achieve increas-
                                         graph (SH), a novel knowledge representation model that
                                                                                                           ingly impressive feats that suggest deeper understand-
                                         is intrinsically recursive and accommodates the natural
                                                                                                           ing.
                                         hierarchical richness of natural language. The SH model
                                                                                                              Presently, there is a strong research focus on the lat-
                                         is hybrid in two senses. First, it attempts to combine the
                                                                                                           ter, and for good reason. Among adaptive models, deep
                                         strengths of ML and symbolic approaches. Second, it is
                                                                                                           neural networks, for one, managed to jointly learn and
                                         a formal language representation that reduces but tol-
                                                                                                           improve performance in classic NLP tasks such as part-
                                         erates ambiguity and structural variability. We will see
                                                                                                           of-speech tagging, chunking, named-entity recognition,
                                         that SH enables simple yet powerful methods of pattern
                                                                                                           and semantic role-labeling [as early as 19]. In other
                                         detection, and features a good compromise for intelligi-
                                                                                                           cases, modern ML enabled methods that did not ex-
                                         bility both for humans and machines. It also provides
                                                                                                           ist before e.g., estimation of semantic similarity using
                                         a semantically deep starting point (in terms of explicit
                                                                                                           word embeddings [45]. More recently, Bidirectional En-
                                         meaning) for further algorithms to operate and collabo-
                                                                                                           coder Representations from Transformers (BERT) have
                                         rate on. We show how modern NLP ML-based building
                                                                                                           shown that pre-trained general models can be fine-
                                         blocks can be used in combination with a random for-
                                                                                                           tuned to achieve state-of-the-art performance in specific
                                         est classifier and a simple search tree to parse NL to SH,
                                                                                                           language understanding tasks such as question answer-
                                         and that this parser can achieve high precision in a diver-
                                                                                                           ing and language inference [21]. Nonetheless, symbolic
                                         sity of text categories. We define a pattern language rep-
                                                                                                           methods possess several proper and important features,
                                         resentable in SH itself, and a process to discover knowl-
                                                                                                           namely that they can offer human-readable knowledge
                                         edge inference rules. We then illustrate the efficiency of
                                                                                                           representations of knowledge, as well as language under-
                                         the SH framework in a variety of tasks, including con-
                                                                                                           standing through formal and inspectable rule-based log-
                                         junction decomposition, open information extraction,
                                                                                                           ical inference.
                                         concept taxonomy inference and co-reference resolu-
                                         tion, and an applied example of claim and conflict anal-             Why do we observe this apparent trade-off between
                                         ysis in a news corpus.                                            openness and adaptivity? Initial approaches to NLP were
                                                                                                           of a symbolic nature, based on rules written by hand,
                                                                                                           or in algorithms akin to the ones that are used for pro-
                                         Keywords: natural language understanding; knowledge               gramming language interpreters and compilers, such as
                                         representation; information extraction; inference sys-            recursive descent parsers. It became apparent that the
                                         tems; explainable artificial intelligence; hypergraphs            diversity of grammatical constructs that can be found in
                                                                                                           natural language is too large to be tackled in such a way.
                                                                                                           The problem is compounded by the frequent use of un-
                                           * menezes@cmb.hu-berlin.de (   )                                grammatical constructs that are nevertheless frequent in
                                           † roth@cmb.hu-berlin.de                                         real-world language usage (e.g. simple mistakes, neol-


                                                                                                       1
ogisms, slang). In other words, content in natural lan-            ual words, and has more refined elaborations, e.g. with
guage is generated by actors that are much more com-               Bayesian regularization [47]. Other methods preserve
plex, and also more error-prone and error-tolerant than            the level of words: such is the case with term and pattern
conventional algorithms. ML is a natural fit for this type         extraction (i.e., discovering salient words through the use
of problem and, as we mentioned, vastly surpasses the              of helper measures like term frequency–inverse docu-
capabilities of human-created symbolic systems in a va-            ment frequency (TF-IDF) [60]), so-called “Named Entity
riety of tasks.                                                    Recognition” [50] (used to identify people, locations, or-
    We suggest however that there is a “hidden rela-               ganizations, and other entities mentioned in corpuses,
tionship” between explicit symbolic manipulation rules             for example in news corpora [22] or Twitter streams [57])
and modern ML: the latter can be seen as a form of                 and ad-hoc uses of conventional computer science ap-
“automatic programming” through large-scale statisti-              proaches such as regular expressions to identify chunks
cal learning processes, that amount to the generation of           of text matching against a certain pattern (for example,
highly complex programs through adaptive pressure in-              extracting all p-values from a collection of scientific arti-
stead of human programmers’ efforts. It does not matter            cles [17]). Another strand of approaches operates at the
if it is gradient descent on a multi-layered network topol-        level of word sets, including those geared at topic detec-
ogy, or something more prosaic like entropy reduction              tion (such as co-word analysis [37], Latent Dirichlet Allo-
in a decision tree, it is still program generation through         cation (LDA) [12] and TextRank [44], used to extract the
adaptation. The capability of these methods to generate            topics addressed in a text) or used for relationship ex-
such complex programs is what allows them to tackle the            traction (meant at deriving semantic relations between
complexities of NL, but it is also this very complexity that       entities mentioned in a text, e.g., is(Berlin, City)) [4]. Re-
makes them opaque.                                                 cent advances in embedding techniques have also made
                                                                   it possible to describe topics extensionally as clusters of
   We can thus imagine a double dichotomy
                                                                   documents in some properly defined space [5, 33].
open/opaque – strict/adaptive. We argue that existing
approaches generally fall into either the open-strict or              Overall, these techniques provide useful approaches
opaque-adaptive categories. A few approaches have                  to analyze text corpora at a high level, for example, with
ventured into the open-adaptive domain [7, 40] and our             regard to their main entities, relationships, sentiment,
contribution aims at significantly expanding this direc-           and topics. However, there is limited support to detect,
tion. Before discussing our approach, let us consider              for instance, more sophisticated claim patterns across
why open-adaptive is a desirable goal. The work we                 a large volume of texts, what recurring statements are
present here was performed in the context of a computa-            made about actors or actions, and what are the qual-
tional social science (CSS) research team, where NLP is a          itative relationships among actors and concepts. This
scientific instrument capable of assisting in the analysis         type of goal, for example, extends semantic analysis to a
of text corpora that are too vast for humans to study              socio-semantic framework [58] which also takes into ac-
in detail. We argue that further progress in the study             count actors who make claims or who are the target of
of socio-technical systems and their dynamics could                claims [22].
be enabled by open-adaptive scientific instruments for               It is also particularly interesting to consider the
language understanding.                                            model of knowledge representation that is implicitly
   In current CSS research, the more common ap-                    or explicitly associated with the various NLP/text min-
proaches aim to transform natural language documents               ing/information extraction approaches. To illustrate,
into structured data that can be more easily analyzed              on one extreme we can consider traditional knowledge
by scholars and are referred to by a variety of umbrella           bases and semantic graphs, which are open in our sense,
terms such as “text mining” [69], “automated text anal-            but also limited in their expressiveness and depth. On
ysis” [29] or “text-as-data methods” [77]. They exhibit            the other, we have the extensive knowledge opaquely en-
a wide range of sophistication, from simple numerical              coded in neural network models such as BERT or GPT-
statistics to more elaborate ML algorithms. Some meth-             2/3 [e.g. 15]. Beyond the desirability of open knowledge
ods indeed rely essentially on scalar numbers, for in-             bases for their own sake, we propose that a language rep-
stance by measuring text similarity (e.g., with cosine dis-        resentation that is convenient for both humans and ma-
tance [64]) or attributing a valence to text, as in the case       chines can constitute a lingua franca, through which sys-
of ideological estimation [63] or sentiment analysis [54],         tems of cognitive agents of different natures can coop-
which in practice may be used to appraise the emotional            erate in a way that is understandable and inspectable.
content of a text (anger, happiness, disagreement, etc.)           Such systems could be used to combine the strengths of
or public sentiment towards political candidates in so-            symbolic and statistical inference.
cial media [76]. Similarly, political positions in docu-             The central idea of our approach is the Semantic Hy-
ments may be inferred from so-called “Wordscores” [39]             pergraph (SH), a novel knowledge representation model
– a popular method in political science that also relies           that is intrinsically recursive and accommodates the nat-
on the summation of pre-computed scores for individ-               ural hierarchical richness of NL. The SH model is hybrid


                                                               2
in two senses. First, it attempts to combine the strengths         mantic information that is lost in the graphic represen-
of ML and symbolic approaches. Second, it is a formal              tation, for example the ability to express n-ary relation-
language representation that reduces but tolerates ambi-           ships, propositions about propositions and constructive
guity, and that also reduces structural variability. We will       definitions of concepts.
see that SH enables simple methods of pattern detec-                  A further type of approaches relying on knowledge
tion to be more powerful and less brittle, that it is a good       bases is epitomized by the famous Cyc [36] project, a
compromise for intelligibility both for humans and ma-             multi-decade enterprise to build a general-purpose and
chines, and that it provides a semantically deeper start-          comprehensive system of concepts and rules. It is an im-
ing point (in terms of explicit meaning) for further algo-         pressive effort, nevertheless hindered by the limitations
rithms to operate and collaborate on.                              that we alluded to in the previous section concerning
   In the next section we discuss the state of the art,            the ambiguity and diversity of semantic structures con-
comparing SH to a number of approaches from various                tained in NL, given that it relies purely on symbolic rea-
fields and eras. We then describe the structure and syn-           soning. Cyc belongs to a category of systems that are
tax of SH, followed by an explanation on how modern                mostly concerned with question answering, a different
and standard NLP ML-based building blocks provided                 aim that the one of the work that we propose here, which
by an open source software library [31] can be used in             is more concerned with aiding in the analysis and sum-
combination with a random forest classifier and a sim-             marization of large corpora of text for research purposes,
ple search tree to parse NL to SH. Here we also pro-               especially in the social sciences, while not requiring full
vide precision benchmarks of our current parser, which             disambiguation of meaning nor perfect reasoning or un-
is then employed in the experiments that follow. We at-            derstanding.
tempted to perform a set of experiments of a rather di-               Several other notable knowledge bases of a similar se-
verse nature, to gather evidence of SH usefulness in a             mantic graph nature have been developed, some relying
variety of roles, and of its potential to tackle the chal-         on collaborative human efforts to gather ground asser-
lenge that we started by stating in this introduction, and         tions, for example MIT’s ConceptNet [68], ATOMIC [61]
to gather empirical insights. One important language               from the Allen Institute, or very rigorous scholarly ef-
understanding task is information extraction from text.            forts of annotation, as is the case with WordNet [46]
One formulation of such a task that attracts significant           and its multiple variants, or relying on wiki-like plat-
attention is that of Open Information Extraction (OIE)             forms such as WikiData [75], or mining relationship
— the domain-free extraction from text of tuples (typ-             from Wikipedia proper, as is the case with DBPedia [6],
ically triplets) representing semantic relationships [24].         and more recently a transformer language model has
We will show that a small and simple set of SH patterns            been proposed to automatically extend common-sense
can produce competitive results in an OIE benchmark,               knowledge bases [14]. We envision that such general-
when pitted against more complex and specialized sys-              knowledge bases could be fruitfully integrated with SHs
tems in that domain. We will demonstrate concept tax-              for various purposes, but such endeavours are beyond
onomy inference and co-reference resolution, followed              the scope of this work. We are instead interested in
by claim and conflict identification in a database of news         demonstrating what can be achieve by going beyond
headers. We will show how SH can be used to generate               such non-hypergraphic appraches.
semantically rich visual summaries of text.

                                                                   Hypergraphic approaches to knowledge representa-
2 Related Work                                                     tion. Hypergraphs have been proposed already in the
                                                                   1970s as a general solution for knowledge representa-
Knowledge bases. As a knowledge representation for-                tion [13]. More recently, Ben Goertzel produced simi-
malism, it is interesting to compare SH with traditional           lar insights [28], and in fact included an hypergraphic
approaches. Let us start with triplet-based ones. For              database called AtomSpace as the core knowledge rep-
example, the Semantic Web [10, 62] community tends                 resentation of his OpenCog framework [30], an attempt
to use standards such as RDFa [1], which represent                 to make Artificial General Intelligence emerge from the
knowledge as subject-predicate-object expressions, and             interaction of a collection of heterogeneous system. As
are conceptually equivalent to semantic graphs [3, 66]             is the case with Cyc, the goals of OpenCog are however
(similarly, a particular type of hypergraph has been used          quite distinct from the aim of our work.
in [16] to represent tagged resources by users, yet this              A model that shares similarities with ours but purely
also reduces to fixed triplet conceptualization). Despite          aims at solving a meaning matching problem is that
their usefulness for simple cases, such approaches can-            of Abstract Meaning Representation (AMR) [7]. AMR is
not hope to match the semantic sophistication of what              based on PropBank verbal propositions and their ar-
can be conveyed with open text. Binary relationships               guments [53], ensuring that all such meaning struc-
and lack of recursion limit the expressive power of se-            tures can be represented. SH completeness is based in-
mantic graphs, and we sill see how SHs can represent se-           stead on Universal Dependencies [52], ensuring instead


                                                               3
that all cataloged grammatical constructs can be repre-             on 8 types) is much simpler than the diversity of gram-
sented. AMR’s goal is to purely abstract meaning, while             matical roles contained in a typical set of dependency la-
SH accommodates the ambiguity of the original NL ut-                bels (such as Universal Dependencies), and we will also
terances, bringing several important benefits: it makes             provide empirical evidence that SHs are not isomorphic
their computational processing tractable in further ways,           to DPTs.
tolerates mistakes better and preserves communication                  In the realm of OIE, one approach in particular with
nuance that would otherwise be lost. Furthermore, it                which our work shares some similarities is that of learn-
remains open to structures that may not be currently                ing open pattern templates [40]. These pattern templates
envisioned. Parsing AMR to NL is a particularly hard                combine at the same symbolic level dependency parse
task and, to our knowledge, there is currently no parser            labels and structure, part-of-speech tags, explicit lexical
that approaches the capabilities of what we will demon-             constraints and higher-order inferences (e.g. that some
strate in this work. In part, this is a practical problem:          term refers to a person), to achieve sophisticated lan-
we will see how we can take advantage of intermediary               guage understanding in the extraction of OIE tuples, be-
NLP tasks that are well studied and developed to achieve            ing able to extract relations that are not only of a verbal
NL to SH parser. Doing the same for AMR requires the                nature, and demonstrating sensitivity to context. The
construction of training data by extensive annotation ef-           work we will present does not attempt to directly com-
forts by humans. It could be argued that this is still a            bine diverse linguistic features at the service of a spe-
preferable goal, no matter how distant, given that AMR              cific language understanding task. Instead, we propose
removes all ambiguity from statements. Here we point                to use such features to aid in the translation of NL into
out that this aspect of AMR is also a downside, firstly be-         a structured representation, which relies by comparison
cause it makes all failures of understanding catastrophic           on a very simple and uniform type system, and from
(we will see how this is not the case for SH), and secondly         which complex NL understanding tasks become easier,
because NL is inherently ambiguous. It is often the case            and that is of general applicability to a diversity of such
that even human beings cannot fully resolve ambigui-                tasks, while remaining fully readable and understand-
ties, or that an ambiguous statement gains importance               able by humans. Furthermore, it defines a system of
later on, with more information. We aim to define SH as             knowledge representation in itself, that is directly fo-
a lingua franca for the collaboration of human an algo-             cused on meaning instead of grammar.
rithmic actors of several natures, a less rigid goal than the
one embodied by AMR.
                                                                    Text mining. We have already covered in the previ-
                                                                    ous section the most commonly used text mining ap-
Free text parsing. A classical NLP task is that of mak-             proaches, while emphasizing the relative lack of sophis-
ing explicit the grammatical structure of a sentence in             tication in understanding text meaning. The need for
the form of a parse tree. A particularly common type of             such sophistication is all the more pregnant for social
such a tree in current use is the Dependency Parse Tree             sciences. On the one hand, qualitative social science
(DPT), based on dependency grammars. We will see that               methods of text analysis do not scale to the enormous
our own parser takes advantage of DPTs (among other                 datasets that are now available. Furthermore, quantita-
high-level grammatical / linguistic features) as interme-           tive approaches allow for other types of analysis that are
diary steps, but it is also interesting to notice that DPTs         enriching and complementary to qualitative research,
themselves can be considered as a type of hypergraphic              yet may simplify extensively the processing in such a way
representation of language [56]. In fact, as we will discuss        that it hinders their adoption by scholars used to the re-
below, they are already employed in various targeted lan-           finement of qualitative approaches. And the more so-
guage understanding tasks in a CSS context.                         phisticated the NLP techniques become, the further they
   From the perspective of hypergraphic representation              tend to be from being used for large-scale text analy-
of language, the fundamental difference between DPTs                sis purposes. Indeed, these systems are fast and accu-
and SHs is that the former aims at expressing the gram-             rate enough to form a starting point for more advanced
matical structure of language, while the latter its seman-          computer-supported analysis in a CSS context, and they
tic structure, in the simplest possible way that enables            enable approaches that are substantially more sophis-
meaning extraction in a principled and predictable way.             ticated than the text mining state of the art discussed
In contrast to the ad-hoc nature of information extrac-             above. Yet, the results of such systems may seem rela-
tion from DPTs, we will see that SHs structure NL in a way          tively simplistic compared to human-level understand-
akin to functional computer languages, and allow for ex-            ing of natural language.
ample for a generic methodology of extracting patterns.                The literature already features some works which at-
The expressive power of such patterns will be demon-                tempt at going beyond language models based on word
strated in several ways, namely by demonstrating com-               distributions (such as bags of words, co-occurrence clus-
petitive results in a standard Open Information Extrac-             ters, or so-called “topics”) or triplets. For instance, State-
tion task. We will see that the type system of SHs (relying         ment Map [49] is aimed at mining the various viewpoints


                                                                4
expressed around a topic of interest in the web. Here              lowing for concepts constructed from other concepts as
a notion of claim is employed. A statement provided                well as statements about statements, and on the other
by the user is compared against statements from a cor-             hand, it can express n-ary relationships. We will see how
pus of text extracted from various web sources. Text               a hypergraphic formalism provides a satisfactory struc-
alignment techniques are used to match statements that             ture for NL constructs.
are likely to refer to the same issue. A machine learn-               While a graph G = (V, E ) is based on a vertex set V
ing model trained over NLP-annotated chunks of text                and an edge set E ⊂ V × V describing dyadic connec-
classifies pairs of claims as “agreement”, “conflict”, “con-       tions, a hypergraph [8, 9] generalizes such structure by
finement” and “evidence”. More broadly, the subfield               allowing n-ary connections. In other words, it can be de-
of argumentation mining [38] also makes extensive use              fined as H = (V, E ), where V is again a vertex set yet E
of machine learning and statistical methods to extract             is a set of hyperedges (e i )i ∈1..M connecting an arbitrary
portions of text corresponding to claims, arguments and            number of vertices. Formally, e i = {v 1 , ...v n } ∈ E = P (V ).
premises. These approaches generally rely on surface               We further generalize hypergraphs in two ways: hyper-
linguistic features, there is however an increasing trend          edges may be ordered [23] and recursive [32]. Ordering
of dealing with structured and relational data. Already in         entails that the position in which a vertex participates
2008, [73] proposed a system to extract binary semantic            in the hyperedge is relevant (as is the case with directed
relationships from Dutch newspaper articles. A recent              graphs). Recursivity means that hyperedges can partici-
work [59] presents a system aimed at analysing claims in           pate as vertices in other hyperedges. The corresponding
the context of climate negotiations. It leverages depen-           hypergraph may be defined as H = (V, E ) where E ⊂ E V
dency parse trees and general ontologies [70] to extract           the recursive© set of all possible hyperedges generated
                                                                   by V : E V = (e i )i ∈{1..n} | n ∈ N, ∀i ∈ {1..n}, e i ∈ V ∪ E V . In
                                                                                                                                   ª
tuples of the form: 〈actor, predicate, negotiation_point〉
where the actors are stakeholders (e.g., countries), the           this sense, V configures a set of irreducible hyperedges
predicates express agreement, opposition or neutrality             of size one i.e., atomic hyperedges which we also de-
and the negotiation point is identified by chunk of text.          note as atoms, similarly to semantic graphs. From here
Similarly, in another recent work [74], parse trees are            on, we simply call these recursive ordered hyperedges as
used to automatically extract source-subject-predicate             “hyperedges”, or just “edges”, and we denote the corre-
clauses in the context of news reporting over the 2008-            sponding hypergraph as a “semantic hypergraph”.
2009 Gaza war, and used to show differences in citation               Let us consider a simple example, based on a set V
and framing patterns between U.S. and Chinese sources.             made of four atoms: the noun “(berlin)”, the verb “(is)”,
   These works help demonstrate the feasibility of using           the adverb “(very)” and the adjective “(nice)”. They may
parse trees and other modern NLP techniques to iden-               act as building blocks for both hyperedges “(is berlin
tify viewpoints and extract more structured claims from            nice)” and “(very nice)”. These structures can further be
text. Being a step forward from pure bag-of-words analy-           nested: the hyperedge “(is berlin (very nice))” represents
sis, they still leave out a considerable amount of informa-        the sentence “Berlin is very nice”. It illustrates a basic
tion contained in natural language texts, namely by rely-          form of recursivity.
ing on topic detection, or on pre-defined categories, or
on working purely on source-subject-predicate clauses.             3.2 Syntax
We propose to introduce a more sophisticated language
model, where all entities participating in a statement are         In a general sense, the hyperedge is the fundamental uni-
identified, where entities can be described as combina-            fying construct that carries information within the SH
tions of other entities, and where statements can be enti-         formalism. We further introduce the notion of hyper-
ties themselves, allowing for claims about claims, or even         edge types, which simply describe the type of construct
claims about claims about claims. The formal backbone              that some hyperedge represents: for instance, concepts,
of this model consists of an extended type of hypergraph           predicates or relationships, as in the above examples —
that is both recursive and directed, thus generalizing se-         respectively (berlin), (is) and (is berlin nice). We exten-
mantic graphs and inducing powerful representation ca-             sively detail hyperedge types and their role in the next
pabilities.                                                        subsections. For now, it is enough to know that predi-
                                                                   cates, in particular and for instance, belong to a larger
                                                                   family of types that are crucial for the construction of hy-
3 Semantic hypergraphs – structure                                 peredges and that we call connectors. In this regard, se-
                                                                   mantic hypergraphs rely on a syntactic rule that is both
  and syntax
                                                                   simple and universal: the first element in a non-atomic
                                                                   hyperedge must be a connector.
3.1 Structure
                                                                      In effect, a hyperedge represents information by com-
The SH model is essentially a recursive, ordered hyper-            bining other (inner) hyperedges that represent informa-
graph that makes the structure contained in natural lan-           tion. The purpose of the connector is to specify in which
guage (NL) explicit. On one hand, NL is recursive, al-             sense inner hyperedges are connected. Naturally, it can


                                                               5
be followed by one or more hyperedges which play the                 As we shall see, these machine-oriented codes remove
role of arguments with respect to the connector. As hy-           ambiguity, facilitate automatic inference and computa-
peredges, if they are not atoms, they must also start with        tions. The full list of types as well as their codes and pur-
a connector themselves, in a recursive fashion.                   poses can be seen in table 1.
   We illustrate this on the hyperedge (is berlin (very
nice)): here, (is) is a predicate playing the role of con-        Connectors The second and last role that atoms can
nector while (berlin) and (very nice) are arguments of the        play is the role of connector. We then have five types of
initial hyperedge. (berlin) is an atomic hyperedge, while         connectors, each one with a specific function that relates
(very nice) is a hyperedge made of two elements: the con-         to the construction of specific types of hyperedges.
nector, (very), an atomic hyperedge, and an argument,                The most straightforward connector is the predicate,
(nice), also an atomic hyperedge. Both cannot be decom-           whose code is “P”. It is used to define relations, which are
posed further.                                                    frequently statements. Let us revisit a previous example
   Readers who are familiar with Lisp will likely                 with types:
have noticed that hyperedges are isomorphic to S-
                                                                                     (is/P berlin/C nice/C)
expressions [42]. This is not purely accidental. Lisp
is very close to λ-calculus, a formal and minimalist              The predicate (is/P) both establishes that this hyperedge
model of computation based on function abstraction                is a relation between the entities following it, and gives
and application. The first item of an S-expression                meaning to the relation. This is isomorphic to typical
specifies a function, the following ones its arguments.           knowledge graphs [3, 66] where (berlin) and (nice) would
One can think of a function as an association between             be connected by an edge labeled with (is).
objects. Albeit hyperedges do not specify computations,
connectors are similar to functions at a very abstract               The modifier type (“M”) applies to one (and only one)
level, in that they define associations. The concepts of          existing hyperedge and defines a new hyperedge of the
“race to space” and “race in space” are both associated to        same type. In practice, as the name indicates, it modi-
the concepts “race” and “space”, but the combination of           fies things and can be applied to concepts, predicates or
these two concepts yields different meaning by applica-           other modifiers, and also to triggers, a type that we will
tion of either the connector “in” or “to”. For this reason,       subsequently address. For concepts, a typical case is ad-
λ-calculus has also been applied to dependency parse              jectivation, e.g.:
trees in the realm of question-answering systems [56].
                                                                                       (nice/M shoes/C)

                                                                  Note here that “nice” is being considered as a modifier,
3.3 Types
                                                                  while “nice” was a concept in the previous case: this is
We now describe a type system that further clarifies the          due to the fact that (nice/M) and (nice/C) refer to two
role each entity plays in a hyperedge. In all, we distin-         distinct atoms which share the same human-readable la-
guish 8 types, the smallest set we could find that appears        bel, “nice”. To illustrate modification of predicates, let us
to cover virtually all possible information representation        revisit a previous example, but suppose that we declare
roles cataloged in the Universal Dependencies. We first           that Berlin is not nice. Then we can apply a modifier to
present the types that atoms may have and discuss their           the predicate, such as (not/M), so that:
use in constructing higher-order entities. We then show
                                                                                ((not/M is/P) berlin/C nice/C)
how hyperedge types are recursively inferable from the
types of the connector and subsequent arguments.                  Finally, modifiers may modify other modifiers:

                                                                                  ((very/M nice/M) shoes/C)
Atomic concepts. The first, simplest and most funda-
mental role that atoms can play is that of a concept. This        The builder type (“B”) combines several concepts to cre-
corresponds to concepts that can be expressed as a sin-           ate a new one. For example, atomic concepts (capital/C)
gle word in the target language, for example “apple”; they        and (germany/C) can be combined with the builder atom
are labeled by this human-readable string, as could be            (of/B) to produce the concept of “capital of Germany”:
guessed from the previous subsection.
                                                                                 (of/B capital/C germany/C)
   This defines an eponymous type, “concept”. The
nomenclature we propose further indicates the type of             A very common structure in English and many other lan-
an atom by appending a more machine-oriented code                 guages is that of the compound noun e.g., “guitar player”
after this label and a slash (/). For concepts, this code         or “Barack Obama”. To represent these cases, we intro-
is “C”:                                                           duce a special builder atom that we call (+/B). Unlike
                                                                  what we have seen so far, this is an atom that does not
                        (apple/C)                                 correspond to any word, but indicates that a concept is


                                                              6
 Code Type               Purpose                                       Example                          Atom Non-atom
   C      concept        Define atomic concepts                        apple/C                            ×          ×

  P       predicate   Build relations                                  (is/P berlin/C nice/C)             ×          ×
  M       modifier    Modify a concept, predicate, modifier,           (red/M shoes/C)                    ×          ×
                      trigger
   B      builder     Build concepts from concepts                     (of/B capital/C germany/C)         ×
   T      trigger     Build specifications                             (in/T 1994/C)                      ×
   J      conjunction Define sequences of concepts or rela-            (and/J meat/C potatoes/C)          ×
                      tions
   R      relation       Express facts, statements, questions, or- (is/P berlin/C nice/C)                            ×
                         ders,...
   S      specifier      Relation specification (e.g. condition, (in/T 1976/C)                                       ×
                         time,...)

Table 1: Hyperedge types with use purposes and examples. Connector types are emphasized with a gray background.
The rightmost columns specify whether this type may be encountered in atomic or non-atomic hyperedges.


formed by the compound of its arguments; it is neces-             argument, and the hyperedge in which they participate
sary to render such compound structures. The previous             has the type of the single argument of the modifier. For
examples can be represented respectively as (+/B gui-             example, the hyperedge (northern/M germany/C) is a con-
tar/C player/C) and (+/B barack/C obama/C).                       cept (C), and (not/M is/P) is a predicate (P).
                                                                     Table 2 lists all type inference rules and their re-
   Conjunctions (“J”), like the English grammatical con-
                                                                  spective requirements. They also induce syntactic con-
struct of the same name, join or coordinate concepts or
                                                                  straints which close the SH type system.
relations:
                                                                     We may now introduce the two last types of our type
               (and/J meat/C potatoes/C)                          system, relation (R) and specifier (S), which only concern
 (but/J (likes/P mary/C meat/C) (hates/P potatoes/C))             non-atomic hyperedges: they are always defined as the
                                                                  result of a composition of hyperedges. Relations are typ-
We also introduce a special conjunction symbol, (:/J),            ically used to state some fact (even though they can also
to denote implicit sequences of related concepts. For             be used to represent questions, orders and other things).
example, the phrase: “Freud, the famous psychiatrist”,            (is/P Berlin/C nice/C) is an obvious example of relation.
would be represented as:                                          In our context, they thus turn out to be a crucial hyper-
                                                                  edge type. Specifiers are types that play a more peripheral
       (:/J freud/C (the/M (famous/M psychiatrist/C)))
                                                                  role, in the proper sense, in that they are supplemental
                                                                  to relations. Specifiers are produced by triggers. For ex-
   The remaining case, triggers (T), concerns additional          ample, the trigger “(in/T)” can be used to construct the
specifications of a relationship, for example conditional         specification: (in/T 1976/C). Specifications, as the name
(“We go if it rains.”), or temporal (“John and Mary trav-         implies, add precisions to relations e.g., when, where,
eled to the North Pole in 2015”), local (“Pablo opened a          why or in which case something happened.
bar in Spain”), etc.:

       (opened/P pablo/C (a/M bar/C) (in/T spain/C))              3.4 Argument roles
                                                                  We introduce a last notion that we employ to make
Hyperedge type inference. Atomic types are entirely
                                                                  meaning more explicit: argument roles for builders and
covered by these six types, of which three exclusively
                                                                  predicates. They are represented as sequences of char-
concern atoms (builders, triggers and conjunctions). We
                                                                  acters that indicate the role of the respective arguments
already hinted at the fact that non-atomic hyperedges
                                                                  following such connectors.
also have types. These are implicit and inferable from the
types of the connector and its arguments. Given, for ex-
ample, that (germany/C) is an atom of type concept (C),           Concept builders. Given a concept hyperedge, a key
the hyperedge (of/B capital/C germany/C) is also a con-           issue is that of inferring its main concept, i.e. the con-
cept, and this can be inferred from the fact that its con-        cept that can be assumed to be its hypernym. Beyond
nector is of type builder (B). Builders need to be followed       the simple case of atoms, concept hyperedges may only
by at least two concepts. Modifiers (M) only accept one           be formed by connectors that are either modifiers or


                                                              7
          Element types → Resulting type                                          Role                    Code
          (M x)                         x                                         active subject             s
          (B C C+)                      C                                         passive subject            p
          (T [CR])                      S                                         agent (passive)            a
          (P [CRS]+)                    R
                                                                                  subject complement         c
          (J x y’+)                     x
                                                                                  direct object              o
                                                                                  indirect object            i
Table 2: Type inference rules. We adopt the notation
                                                                                  parataxis                  t
of regular expressions: the symbol + is used to de-
                                                                                  interjection               j
note one or more entities with the type that precedes it,
                                                                                  specification              x
while square brackets indicate several possibilities (for
                                                                                  relative relation          r
instance, [CR]+ means “at least one of any of both C or
R” types). x means any type: (M x) is of type x.
                                                                               Table 3: Predicate argument roles.

builders. When the connector is a modifier, finding the
hypernym is admittedly trivial. When the connector is              due to the flexibility of NL in this regard, and to the fact
a builder, it is often possible to infer the main concept          that the presence of a certain role after a predicate is of-
among the arguments. There are only two possible roles:            ten optional.
“main” (denoted by m) and “auxiliary” (denoted by a).                 There are admittedly more possible roles than for
For example:                                                       builders. They are shown in table 3. Once again, this set
                                                                   is the result of an effort to cover all grammatical cases
                (+/B.am tennis/C ball/C)                           listed in the Universal Dependencies in the most suc-
                                                                   cinct way possible. Most of them (in fact, the first 8 in the
The argument role annotation “.am” indicates that ball/c           table) directly correspond to generic grammatical roles
is the main concept in the construct, meaning that                 of the same name. Of these, the first 6 are by far the
(+/B.am tennis/C ball/C) is a sort of ball/c — the main            most frequent. Specifications were already discussed in
concept is a hypernym of the whole construct.                      the previous subsection (3.3), and their purpose as hy-
   With compound nouns ((+/B) builder), we simply                  peredges coincides with their role when participating in
make use of part-of-speech and dependency labels to in-            relations: as an additional specification to the relation
fer the main concept. Another common situation where               (temporal, conditional, etc.). Finally, a relative relation
finding roles is quite trivial is the case of builders de-         is a nested relation, that acts as a building block of the
rived from a proposition, such as (of/B), which express            outer relation that contains it. We will make extensive
a relationship between the arguments. For example, in              use of this later, to identify what is being claimed by a
(of/B.ma capital/C germany/C), the main concept is (cap-           given actor.
ital/C). “Capital of Germany” is thus a type of capital. In
English and many other languages, it is always the case
that the main concept is the first argument after a builder
derived from a proposition.
                                                                   4 Translating NL into SH
                                                                   We now discuss the crucial task of translating NL into
Predicates. Predicates can induce specific roles that              this SH representation. This can, of course, be framed
the following arguments play in a relation. The need for           as a conventional supervised ML task. A difficulty arises
argument roles in relations arises from cases where the            from the lack of training data. SH is a novel repre-
role cannot be inferred from the type of the argument.             sentation, and the effort necessary to annotate a suffi-
For example, the same concept could participate in a re-           ciently large amount of text to train an NL to SH transla-
lation as a subject or as an object. Consider for instance         tor from scratch is far from trivial. We were motivated
the sentence “John gave Mary a flower”, represented as:            to look for an alternative, and we hypothesized that it
                                                                   would be much easier to infer the SH representation
       (gave/P.sio john/C mary/C (a/M flower/C))
                                                                   from grammatically-enriched representations than from
In this relation, the argument role string “sio” indicates         raw text. We will show that this indeed appears to be the
that the three arguments following the predicate respec-           case.
tively play the roles of subject, indirect object and direct          We propose a two-staged approach. The first (α-stage)
object. This relation involves three concepts united by            is a classifier that assigns a type to each token in a given
the predicate that represents the act of giving, but with-         sentence. The second (β-stage) is a search tree-based al-
out the argument roles, who the giver is, who the receiver         gorithm that recursively applies the rules in table 2 to
is, and what object is being given, would remain unde-             impose the hypergraphic structure on the sequence of
fined. Relying on ordering would not be enough, both               atoms produced by the α-stage. This restricts the ML


                                                               8
part of the process to the α-stage, making it a trivial clas-                  are reported in [67] to be 0.97 for the fine grained part-of-
sification problem.                                                            speech tagger (i.e., guessing the OntoNotes tag), 0.92 for
                                                                               unlabeled dependencies (i.e., guessing the head of each
                                                                               token) and 0.90 for labeled dependencies (i.e., the head
4.1 α-stage                                                                    and the label).
The classification categories correspond to the set of the                        Let us refer to the former as TAG, and to the latter as
six atomic types shown in table 1, with one additional                         POS. We can also consider the most common words in
category for tokens that should be discarded (typically                        the corpus. We consider as features the sets of 15, 25, 50
punctuation). The open question is the feature set. We                         and 100 most common words (WORD15, WORD25 and
will see how, operating on the previous assumption re-                         so on). Further features indicate if a token corresponds
garding grammatical annotation, we use spaCy1 – a pop-                         to some punctuation symbol, if it is at the root of the de-
ular NLP tool – to generate appropriate features.                              pendency parse trees, if it has left or right children in this
   Using this library we perform segmentation of text                          same tree, and finally its shape in terms of capitalization
into sentences, followed by tokenization and annota-                           (e.g. the shape of the word “Alice” is Xxxxx). Then, we
tion of tokens with parts-of-speech, dependency labels                         establish three types of relative tokens: the ones that ap-
and named entity categories. In short, we deploy the                           pear directly after or before the current one in the sen-
full arsenal of off-the-shelf NLP tasks that come avail-                       tence, if they exist, and the one that is the parent of the
able with spaCy. In this work we restrict ourselves to the                     current one in the dependency parse tree, if it exists.
English language and we use the “en_core_web_lg-2.0.0”                         For each one of these tokens, all the previous features
language model.                                                                are also applied (for example, the UD part-of-speech of
   We collected randomly selected texts in English from                        the dependency head is HPOS, and the part-of-speech of
five categories: fiction (5 books, 87738 sentences) and                        the subsequent word in the sentence is POS_AFTER). We
non-fiction books (5 books, 51597 sentences), news (10                         thus have 33 candidate features in total. All of these fea-
articles, 532 sentences), scientific articles (10 articles,                    tures are categorical, and we employ one-hot encoding
3467 sentences) and Wikipedia articles (10 articles, 2888                      to feed them to the decision trees.
sentences). From these we selected 60 random sen-
tences in each category, thus a total of 300 sentences rep-                    Feature selection. We tested two approaches for fea-
resenting 6936 tokens. An interactive computer script                          ture selection: a very simple genetic algorithm (GA) and
was used to aid in the process of manually annotating                          iterative ablation. For the GA, we encoded features as
each word of these sentences with one of the alpha cate-                       bits (acting as switches to specify which features be-
gories i.e., atomic types. These were used to train a ran-                     long to the set). We used mutation only (bit-flip with a
dom forest classifier. For this purpose we employed the                        probability of .05), a population of 100, and parent se-
one included with scikit-learn (version 0.23.2), a widely                      lection through a tournament of 3. Search stopped at
used ML package. We did not perform any hyperparam-                            100 generations without improvement. The fitness func-
eter tuning, and used the default parameters set by this                       tion was the mean of 5 evaluations of the accuracy of
version of the package. There is possibly room for im-                         the feature set, each with a distinct and randomly se-
provement here. For the aims of this work, we found it                         lected split of the training / testing data. This even-
preferable to avoid introducing potentially confounding                        tually resulted in a set of 15 features: {WORD25, TAG,
factors that could arise from hyperparameter optimiza-                         DEP, HWORD25, HWORD50, HWORD100, HPOS, HDEP,
tion efforts.                                                                  IS_ROOT , NER, WORD_BEFORE15, WORD_BEFORE100,
                                                                               WORD_AFTER15, PUNCT_BEFORE, POS_AFTER}.
Feature definition. We consider an initial set that en-                            The iterative ablation procedure starts with the set of
compasses all the potentially useful features that we                          all candidate features, and 100 runs of the learning algo-
could derive from a standard NLP pipeline such as spaCy.                       rithm are performed, again each run randomly split into
As we mentioned, it provides dependency parse labels                           two-thirds for training and one-third for testing. This
(referred to, from now on, as DEP) and named entity                            provides us with a set of 100 accuracy measurements.
recognition categories (NER). Parts-of-speech are pro-                         The process is then repeated, excluding one feature at at
vided in two flavors: the more extensive OntoNotes tag                         time. The feature that most degrades mean accuracy is
set (version 5) from the Penn Treebank, and the sim-                           excluded. If no feature has a negative impact on accu-
pler Universal Dependencies (UD) part-of-speech tag set                        racy, then the one with the highest p-value (according to
(version 2). Accuracy values for each of these elements                        the non-parametric Kolmogorov–Smirnov test) above a
                                                                               threshold is excluded. The procedure is repeated, ablat-
   1 An open-source library for NLP in Python which includes convo-
                                                                               ing one feature at a time, until no remaining feature ful-
lutional neural network models for tagging, parsing and named entity
                                                                               fills any of the previous two criteria. We performed this
recognition in multiple languages. A relatively recent comparison of
ten popular syntactic parsers found spaCy to be the fastest, with an ac-       procedure with threshold p-values of .05 and .005. The
curacy within 1% of the best one [18]                                          first left us with a set of five features: F5 = {TAG, DEP,


                                                                           9
HDEP, HPOS, POS_AFTER}; the second with three fea-                      Function ApplyPattern(seq, pos, pat )
tures: F3 = {TAG, DEP, HDEP}.                                              Data: A sequence of edges seq, a position in the
   The results of these experiments are shown on the left                         sequence pos and a pattern pat
side of figure 1. As can be seen, all of the three attempts                Result: A sequence of edges with the initial edges
outperform the set of all features. Interestingly, F5 is sig-                       replaced by a single one, if they match the
nificantly better than F3, even at p < .005. The accu-                              pattern.
racy of the GA set falls between that of F3 and F5. We                     if pat matches seq at pos then
                                                                               ed g e ←− reorder matching elements of seq to
performed these experiments not only as an endeavor to
                                                                                align with pat
achieve acceptable accuracy for the experiments that fol-                      seq 0 ←− matching part of seq replaced with
low, but also to obtain empirical evidence regarding the                        ed g e
relationship between SH types and traditional linguistic                       return seq 0
features. We can conclude that SH does not correspond                      else
to some trivial mapping of any single linguistic feature.                      return ∅
For subsequent experiments we will use F5, given that                      end if
it has the best accuracy and still uses a relatively small              end
number of features – something that can make a differ-
                                                                        Function BetaTransformation(seq)
ence regarding the computational effort needed to parse                    Data: A sequence of edges seq
large quantities of text. It is interesting to notice that F3              Result: An edge e
still leads to a higher accuracy than the set of all features,             if |seq| = 1 then
and having only three features, such a classifier could                         return seq[0]
be feasibly implemented in a purely programmatic way.                      end if
A completely human-understandable classification tree                      heu best ←− −∞
could be produced, and also implemented in a very effi-                    seq best ←− ∅
cient way, sacrificing relatively little in terms of accuracy.             for pos = 1 to |seq| do
                                                                                for pat ∈ P at t er ns do
   On the right side of figure 1 we present the accuracy of
                                                                                    seq 0 ←− ApplyPattern(seq, pos, pat )
the classifier by text category, using F5. Here, it is inter-
                                                                                    heu ←− h(seq, pos, pat )
esting to note that the best performing category (fiction)
                                                                                    if seq 0 6= ∅ ∧ heu > heu best then
and also one of the second-best (wikipedia, which is not                                 heu best ←− heu
significantly different from news) are out-of-corpus for                                 seq best ←− seq 0
the training set of the ML model of the underlying lin-                             end if
guistic features. It is remarkable that the accuracies that                     end for
we achieve are comparable and may even surpass the                         end for
values reported by spaCy (see above). In other words,                      if seq best 6= ∅ then
this suggests that, far from accumulating errors down the                       return BetaTransformation(seq best )
stream of the various processing steps, our α stage ap-                    else
pears to even correct upstream errors.                                          return ((:/J) + seq[: 2] ) + seq[2 :]
   It is conceivable that more features become relevant,                   end if
if a larger number of exotic cases becomes available                    end
through larger training corpora. It is also conceivable                Algorithm 1: The β transformation recursively ap-
that larger windows (beyond just previous and next to-                 plies the patterns from type inference rules until
ken) become relevant with larger datasets and more so-                 only the final hyperedge is left.
phisticated ML approaches. Such considerations are be-
yond the scope of this work.

                                                                      how β iteratively constructs a hyperedge, which need not
4.2 β-stage
                                                                      be a proper semantic hyperedge except at the final step.
The β-stage transforms the sequence of atoms of the                   The process starts indeed with an initial hyperedge as the
original sentence, each typed by the α-stage, into a se-              simple sequence of typed atoms of the original sentence.
mantic hyperedge that reflects the meaning of the sen-                At each step, the elements of the currently-formed hy-
tence and respects the SH syntactic rules. In practice,               peredge are scanned from left to right to look for a sub-
this operation amounts to a bottom-up process that ag-                sequence of types that matches the list on the left side
gregates the deeper structures of the sentence into in-               of the type inference rules of table 2, taken as unordered
creasingly complex hyperedges, by recursively combin-                 patterns i.e., up to any reordering. For instance, “capi-
ing them until only a final, well-formed semantic hyper-              tal of Germany” may have been parsed by α as a typed
edge is left.                                                         sub-sequence “capital/C, of/B, germany/C”, which then
   The process for this transformation is formalized in               matches the second pattern (B C C). It may then be rear-
algorithm 1. Let us nonetheless explain in plain words                ranged as such by putting the connector in first position


                                                                 10
Figure 1: Left: accuracy of the α-classifier, comparing several feature sets; all includes all features, GA a features set
obtained with a genetic algorithm, F3 is the outcome of iterative ablation with p < .005 and F5 with p < .05. Right:
accuracy by source text category using F5.


and preserving the order of the remainder of the hyper-             the bottom-up process of the β-transformation. Finally,
edge i.e., “(of/B capital/C germany/C)”, which conforms             if there is still a tie, rules are applied by the order of pri-
to the second inference rule of table 2. Note that, in prac-        ority expressed in table 2, which is empirically organized
tice, we also restrict the second and fifth patterns, i.e.          by decreasing order of the depth at which each respec-
the builder and conjunction patterns, to the minimum                tive structure tends to appear in hyperedges. The special
number of two arguments: respectively (B C C) and (J x              rule for (+/B) is assigned the highest priority.
x 0 ). We find that it fits NL more naturally and thus leads            If no sub-sequence matches, the two first items in the
to more correct parses. Further tasks of knowledge in-              sequence are connected by prepending the special con-
ference might later introduce builder- and conjunction-             junction (:/J), which is meant to convey the most generic
based structures with more arguments. We complement                 and abstract meaning of “these two things are related in
the patterns with one rule that corresponds to the special          the most generic sense”. This captures cases often found
connector (+/B). This extra rule is admittedly needed to            in natural language, such as: “A new era: quantum com-
transform implicit builders (C C) into (+/B C C).                   putation is here.”, which translates to:
   If only one sub-sequence matches, it is transformed                    (:/J (a/M (new/M era/C)) (is/P (quantum/M
into a sub-hyperedge by application of the rule. If two or                          computation/C) here/C))
more sub-sequences match, the β-stage needs to make
a decision on which one to choose and proceed with as                  If the resulting hyperedge entirely conforms to one of
if only one sub-sequence matched. For this case, we use             the type inference rules, the process stops successfully as
a heuristic function (this is function h in algorithm 1).           it managed to form a recursively correct semantic hyper-
This heuristic function relies on the grammatical struc-            edge. Otherwise, the process is reiterated on the newly-
ture of the sentence given by the dependency tree. Our              formed hyperedge. The process is thus guaranteed to
hypothesis is that grammatically connected edges are                converge on a syntactically valid hyperedge, but is of
more likely to belong to the same higher-order edge, so             course not guaranteed to produce the most desirable or
the first criterion of h is to always assign a higher score         correct representation. However, we experimentally ver-
to sub-sequences where all items are directly connected             ify below that, given a correct classification from the α-
in the dependency tree. By “directly connected in the               stage and a correct dependency parse tree, this process
dependency tree”, we mean that all hyperedges contain               consistently leads to the construction of a SH that cor-
one atom/token that is the head or the child of at least            rectly conveys the meaning of the original sentence.
one atom/token in another hyperedge, and that any hy-                  Let us first illustrate the β-stage in figure 2, which pro-
peredge can be reached from any other, following such               vides one example of an entire parsing process (using
grammatical links. In case there is a tie, the heuristic            the F3 feature set for simplicity). In figure 2(c), the re-
function then prefers the sub-sequence that contains the            cursive application of β-transformations to an initial se-
deepest atom/token in the dependency tree – again as-               quence of atoms can be followed. In the first step, we
suming a correlation with SH depth, and thus respecting             can see that the sequence (the/M, capital/C) matches the


                                                               11
Figure 2: (a) Dependency parse tree with dependency labels (green) and fine grained part-of-speech tags (red). (b)
α-stage classification of atom types. (c) β-stage structuring of sentence by iterative application of the patterns from
table 2. A non-selected pattern is greyed-out.


pattern (M C), and the sequence (capital/C, of/B, ger-             year/C) or (multi/M year/C) would be much preferable.
many/C) matches the pattern (B C C+). We thus rely on              However, this partially defective parse is still likely to be
the above-mentioned heuristic function, which causes               useful in the methods that we will discuss in the follow-
(of/B capital/C germany/C) to be preferred to (the/M cap-          ing sections. We also see how different type assignments
ital/C). The reader can verify that selecting the latter at        of the α-classifier can result, in practice, in correct hyper-
this stage would lead to a dead-end. The rest of the SH            edges at the end. We can also use this example to illus-
construction is straightforward.                                   trate another metric that we employ in this evaluation:
                                                                   the relative defect size. This is simple the ratio of the size
Argument roles. Now that the core of the translation of            of the defective part to the size of the entire hyperedge.
NL into SH has been specified, assigning the argument              Size is measured in total number of atoms (at any depth).
roles introduced in Section 3.4 amounts to a trivial trans-           A wrong hyperedge is one where the meaning of the
lation from the dependency labels. Sometimes however,              sentence is completely lost. For example, consider what
the parser may fail to determine an argument’s role, and           would happen if, in the above case, “stressed” was classi-
thus classify it as unknown (that we code “?” for this pur-        fied as a concept instead of a predicate. This also serves
pose).                                                             to illustrate that there is a complex relationship between
                                                                   α-classifier accuracy and overall parser accuracy. Some
4.3 Validation of α and β                                          mis-classifications at the α-stage can still allow for a
                                                                   completely correct parse, while others can lead to catas-
To test the accuracy of the complete translation from              trophic failures or just minor defects. Nonetheless, we
NL to SH, we randomly selected 100 new sentences for               observe on this sample of 500 sentences that a correct
each text category, that were used neither for training            α classification and dependency parse tree always lead
nor testing of the α-classifier. We establish three cate-          to the construction of an SH that preserves the meaning
gories: completely correct hyperedges, hyperedges with             of the sentence. By contrast, a badly-structured depen-
some defect and completely wrong hyperedges. A hyper-              dency tree appears to have a significant negative impact
edge is considered to have a defect if overall meaning is          on the functioning of β, through the heuristic function.
preserved, but some subedge contains a defect. Let us              If this result generalizes, this suggests that, for a given
consider a real example from our dataset. The sentence:            accuracy of the dependency parsing module, increasing
“The scientists – who are part of a multi-year Interna-            the quality of the NL to SH translation principally relies
tional Shelf Study Expedition – stressed their findings are        on improving α and the heuristic function.
preliminary.” was parsed as:
                                                                      We show the results of this evaluation in table 4. It
(stressed/P (:/J (the/M scientists/C) (are/P who/C (of/B           is interesting to notice that “non-fiction” is one of the
   part/C (a/M (+/B (+/B (+/B multi/C -/C) year/C)                 worst performing categories in the α-classifier, but ends
    (+/B international/C (+/B shelf/C (+/B study/C                 up being the best one overall. Likewise, “fiction” is the
     expedition/C)))))))) (are/P (their/M findings/C)              best category at α-stage but ends up being the second
                      preliminary/C))                              worst here. Unsurprisingly, “fiction” sentences tend to
                                                                   be richer in figures of speech and other complexities and
  The hyperedge preserves most of the meaning of the               ambiguities that lead to a higher rate of catastrophic fail-
sentence, but the concept (+/B (+/B multi/C -/C)                   ure. Conversely, “non-fiction” is the category with the
year/C) is not correctly formed. Either (-/B multi/C               most straight-forward sentences. In the “science” cate-


                                                              12
                    Category        Correct     Defect     Wrong         Total   Mean relative defect size
                    Non-fiction     87 (.87)  8 (.08)  5 (.05)            100              .188
                    Wikipedia       81 (.81) 12 (.12) 7 (.07)             100              .190
                    News            77 (.77) 16 (.16) 7 (.07)             100              .147
                    Fiction         79 (.79)  5 (.05) 16 (.16)            100              .140
                    Science         71 (.71) 19 (.19) 10 (.10)            100              .290
                    All            395 (.79) 60 (.12) 45 (.09)            500              .206

                                       Table 4: Global NL to SH parser evaluation.


gory, the difficulties are more related to a variety of un-          5.1 A pattern-matching language
usual technical terms and notations, that lead more to
                                                                     From a text corpus, the NL to SH translation stage at-
defects than catastrophic failures. Overall, we see that
                                                                     tempts to convert each sentence into a hyperedge. In
a high percentage of the texts are correctly translated to
                                                                     practice, all resulting hyperedges are stored in a proper
SH, even in the worst-performing categories.
                                                                     SH database. From there, language understanding tasks
   We only work with English in this article, but support-           may be performed in the form of inferences, which
ing a new language essentially requires to generate a rel-           we define using SH notation with the help of patterns.
atively small number of α-classifier training examples.              Broadly, inference rules and patterns may also be writ-
The rest of the process is currently language-agnostic:              ten as hyperedges.
even though more research would be needed to explore
this issue thoroughly, the fact that we cover all of the Uni-        Variables and patterns. We introduce the concept of
versal Dependency cases gives us good reasons to be-                 variable. A variable simply indicates a placeholder that
lieve that NL to SH translation shall be applicable to any           can match a hyperedge, and then be used to refer to that
language. The software package that we released to im-               hyperedge. Unlike the other atoms we have seen so far,
plement all the ideas discussed in this article includes             variables are represented in capital letters. With vari-
the interactive script that we used to perform this anno-            ables we can define patterns, that can then be matched
tation task ourselves.                                               against other hyperedges. For example, consider the pat-
                                                                     tern:

                                                                                     (is/P.sc SUBJ PROP/C)

                                                                     which matches, for example:
5 Knowledge Inference and Extrac-                                                 (is/P.sc (the/M sky/C) blue/C)
  tion
                                                                        Notice that the variable PROP includes a type code,
                                                                     while the predicate “is/P.sc” features argument roles. If
We are finally in the position to explore the use of SH ex-          type codes or argument roles are added to variables, this
tracted from open text to perform language understand-               simply means that a hyperedge only matches this vari-
ing tasks. First we will discuss how we naturally gener-             able if the types and argument roles also match.
alize SH to represent patterns and inference rules, and                 We introduce a few more notation details for argu-
then we will manually define three such rules to per-                ment roles in patterns. In practice, we often allow the
form conjunction resolution: a very useful and generic               various pattern elements to appear in any order, denot-
task that will be used in every following practical appli-           ing the order-indifferent roles between curly brackets “{
cation discussed in this work, and very likely useful for            }”. The arguments can appear in any order, as long as all
myriad knowledge inference and extraction tasks. Then                of the pattern roles are present. For instance,
we will discuss how we systematized the process of dis-                             (is/P.{sc} SUBJ PROP/C)
covery of useful patterns, and how we use this process to
discover 8 patterns for the purpose of Open Information              would both cover (is/P.sc SUBJ PROP/C) and (is/P.cs
Extraction (OIE), for which an abundant computer sci-                PROP/C SUBJ). Furthermore, it is possible to specify the
ence literature exists where scholars are interested in in-          optional presence of certain arguments by listing them
ferring relations from free text. We will demonstrate the            as “...”, which simply indicates that any number (includ-
expressive power of SH by showing that these simple pat-             ing zero) hyperedges may be present at that point. In a
terns produce competitive results when compared with                 pattern, if the connector indicates argument roles, then
a number of contemporary systems targeted at OIE, us-                any further arguments may be present, unless indicated
ing an external benchmark.                                           otherwise. In case connectors do not indicate argument


                                                                13
roles, “...” can thus be used to indicate that more hyper-             The second rule concerns conjunctions of relations
edges at a certain point are permissible. For instance,              with explicit subjects, for example: “Mary likes astron-
                                                                     omy and Alice plays football.”, which is parsed as:
               (is/P.{sc} SUBJ PROP/C ...)
                                                                        (and/J (likes/P.so mary/C astronomy/C) (plays/P.so
matches “The sky is blue today” and “Today the sky is                                    alice/C football/C))
blue”. It is also possible to denote an undefined sequence
of hyperedges with a variable name by using “X...”, which            is decomposed into: “Mary likes astronomy.” and “Alice
thus refers to the same specific sequence everywhere it              plays football.” i.e.,:
is used.
                                                                                  (likes/P.so mary/C astronomy/C)
   Sequences of alternative arguments roles of which
                                                                                    (plays/P.so alice/C football/C)
anyone of them can be matched once are represented in-
side square brackets. For example, “[sp]” matches either               The third rule makes the subject explicit in situations
a subject or a passive subject, once. Finally, it is possi-          such as “Mary likes astronomy and plays football.” i.e.,
ble to forbid the presence of arguments with a certain
role by listing them after “-”. For example, in the pat-                 (and/J (likes/P.so mary/C astronomy/C) (plays/P.o
tern “(PRED/P.-sp X...)”, arguments with roles “s” or “p”                                    football/C))
are not allowed; it would however match (play/P.o foot-
ball/C).                                                             inferring that “Mary” is the subject from the first relation
                                                                     in the conjunction and applying it to the second one:
                                                                     “Mary plays football.”, resulting in:
Rules. We may now define rules which we denote with
a couple of patterns separated by the symbol “`”, as                              (likes/P.so mary/C astronomy/C)
in “PATTERN1 ` PATTERN2”. This notation indicates                                  (plays/P.so mary/C football/C)
that any hyperedge that contains a hyperedge matching
the left-hand-side PATTERN1 would incur the creation                 In practice, this is done by remembering the last argu-
of a duplicated hyperedge consisting of the matching                 ment with the subject (s) role and applying it to the fol-
portion rewritten according to the right-hand-side ex-               lowing relations that miss a subject.
pressed as PATTERN2. In a sense, these are replacement                  Naturally, these rules have a lot of space for improve-
rules, except that the original hyperedge is preserved.              ment, not making distinctions for conjunctions with
  For example, consider the rule:                                    special meaning (e.g. “but”, “instead”, etc.). Neverthe-
                                                                     less, we will see that they are already quite successful in
     (is/P.sc SUBJ PROP/C) ` (property/P PROP)                       the tasks that we will subsequently present.

which, applied to the above example, produces the infer-
ence:                                                                5.3 Pattern Learning
                                                                     With the help of hypergraphs extracted from corpora of
                    (property/P blue/C)                              open text, it becomes possible to define a systematic pro-
                                                                     cess of discovery of patterns that enable knowledge ex-
  In essence, a rule makes it possible to populate an SH             traction, with a human-in-the-loop. On the left side of
database with new knowledge that is inferred from NL                 figure 3 we present the general template for such a pro-
yet need not, in turn, correspond to an actual sentence.             cess. We will use this template to illustrate both how we
                                                                     discovered the patterns for the OpenIE task as well as
5.2 Conjunction Decomposition                                        the claim and conflict analysis that will be presented in
                                                                     section 5.4, and also how more sophisticated and auto-
Decomposing relations that include conjunctions into                 mated pattern learning systems can be created. The rest
simpler relations not only facilitates OIE tasks, but is also        of figure 3 is a step-by-step illustration on a simple exam-
of general usefulness in knowledge inference tasks. We               ple aimed at generating patterns to detect claims.
show the three rules that we developed manually to per-                 In step (1), a hyperedge is selected from the hyper-
form conjunction decomposition in table 5.                           graph generated from a given training corpus. It can be
   The first rule concerns conjunctions of concepts, such            drawn at random, or by any other criterion adapted to
as “Mary likes books and flowers.”:                                  the pattern-learning task at hand. For instance, if we
                                                                     want to learn patterns typical of claims, we can first fo-
      (likes/P.so mary/C (and/J books/C flowers/C))
                                                                     cus on hyperedges starting with a predicate (*/P) and,
where we generate one relation for each element:                     more precisely, the most frequent ones among them,
                                                                     as a strategy to attain good coverage. We observe that
               (likes/P.so mary/C books/C)                           “says/P” is such a predicate and based on this, we draw
               (likes/P.so mary/C flowers/C)                         (says/P.sr alice/C (are/P.sc dogs/C nice/C)) i.e., “Alice says


                                                                14
 # Rule                                                                             Inferences
   ¡                        ¢
 1 ³*/J ... CONCEPT/C ... ` (CONCEPT/C)´                                                147
            ¡                     ¢       ¡                   ¢
 2 */J ...    PRED/P.{[sp]} X Y... ... ` PRED/P.{[sp]} X Y...                            63
   ³    ¡                     ¢     ¡               ¢   ´  ¡                      ¢
 3 */J */P.{[sp]} SUBJ/* ... ...     PRED/P.-sp X... ... ` PRED/P.{s} SUBJ/* X...        10

   Table 5: Conjunction resolution rules and respective number of inferred hyperedges from the OIE benchmark
                                                         .


                                                         FIRST PASS                                            SECOND PASS



                                                “Alice says dogs are nice.”
           (1) Select hyperedge
                                          (says/P.sr alice/C (are/P.sc dogs/C nice/C))




                                                ACTOR             CLAIM


                                               “Alice says dogs are nice.”
           (2) Human inference
                                         (says/P.sr alice/C (are/P.sc dogs/C nice/C))
                                      ACTOR = alice/C CLAIM = (are/P.sc dogs/C nice/C)




                                          (says/P.sr alice/C (are/P.sc dogs/C nice/C))                  Make pattern more specific:
           (3) Pattern inference
                                                    (*/P.{sr} ACTOR CLAIM)                              (says/P.{sr} ACTOR CLAIM)




                                                    (*/P.{sr} ACTOR CLAIM)                              (says/P.{sr} ACTOR CLAIM)

          (4) Search with pattern               “Bob wants to play chess.”                  “The president says the economy will recover.”
                                         (wants/P.sr bob/C ((to/M play/P.o) chess/C))                    (says/P.sr (the/M president/C)
                                                                                                  ((will/M recover/P.o) (the/M economy/C)))




                                           Wrong match, pattern is too generic:
                                                                                                     ACTOR = (the/M president/C)
           (5) Human feedback                      ACTOR = bob/C                            CLAIM = ((will/M recover/P.o) (the/M economy/C))   ✓
                                            CLAIM = ((to/M play/P.o) chess/C)


Figure 3: Pattern learning template and example with two passes. At the end of the second pass, the pattern (says/P.sr
ACTOR CLAIM) is confirmed to work.


dogs are nice”. Other pattern-learning tasks may natu-                    the human inference. The matching parts are replaced
rally require different selection criteria, and in the next               by the corresponding variables, and the remaining sub-
subsection (5.4) we will provide another and more gen-                    edges are replaced by wildcards, while maintaining type
eral example.                                                             annotations.
   A human is then presented with this hyperedge in step                    Then the process goes back to the whole training hy-
(2), and asked to manually perform an inference. The                      pergraph. Step (4) consists of finding hyperedges that
inference consists of selecting sub-edges and assigning                   match the pattern, so that they can be presented to the
them to variables according to some schema. In the ex-                    human for validation. Then, in step (5), the human can
ample shown, the inference aim is to detect which actors                  simply indicate if these further matches are valid or not.
making claims, and thus identify which “ACTOR” makes                         When a match is not valid, the process can then return
which “CLAIM”.                                                            to step (3) and use this information to refine the pattern.
   Step (3) consists of generalizing the original hyperedge               What is now the most general version of the pattern that
into a pattern with the help of the variable assignments.                 does not match the previously detected incorrect case?
The idea is to create the most generic pattern that fits                    In our work, we used a conventional Jupyter notebook


                                                                   15
directly accessing the programmatic interface of Graph-             becomes:
brain2 , the open-source library that we developed to im-
                                                                                         (*/P.sc */C */C)
plement all of the ideas discussed in this work. Refine-
ments at step (3) were performed manually, testing hy-                The process can continue recursively, further expand-
pothesis on the most general version of a pattern by sim-           ing subedges, for instance:
ply asking Graphbrain to check how many actual hyper-
edges match each attempt, and choosing the one with                              (*/P.sc */C (*/B.ma */C */C))
the highest value. This process can obviously be au-
                                                                       These expansions have to conform to Table 2 (taken in
tomated with a search tree, that attempts a number of
                                                                    reverse order i.e., from a resulting type to its antecedent),
substitutions – more or less generic wildcards, lemma
                                                                    which generally leaves a small number of possibilities.
matching, atom root matching, structural matching, etc.
                                                                    We further introduced some restrictions in the genera-
– at each step, and then uses the training hypergraph
                                                                    tion of such patterns, to focus on simple patterns with
to empirically test them and discover the most generic
                                                                    likely relevance to our task. We limited recursive expan-
one that correctly matches both the positive and neg-
                                                                    sion to depth 2, and only consider relations of sizes 3 or
ative cases known so far. Then, even more sophisti-
                                                                    4 – smaller ones cannot contain triplets, larger ones that
cated possibilities arise, such as the integration with gen-
                                                                    are useful are likely to contain the triplet (with optional
eral knowledge databases (e.g. specifying that a variable
                                                                    extension) within a core that generalizes to patterns with
must be a concept of type “country”, or that a predicate
                                                                    no more than 4 elements. We excluded conjunctions
must be the synonym of a certain action), or the use
                                                                    (these are previously decomposed, as explained in sub-
of auxiliary methods such as semantic proximity with
                                                                    section 5.2), and modifiers. These latter connectors
word2vec-like embeddings, or hybridization with ML al-
                                                                    could certainly be used to improve the OIE task, but en-
gorithms.
                                                                    tail more semantic complexity, and we are more inter-
   Another possible improvement is in the domain of
                                                                    ested in simplicity at this stage. We will focus on mod-
software and user-interface development, allowing for
                                                                    ifiers in a subsequent section. Finally, we allow for the
less technical users to provide inferences and feedback –
                                                                    special builder (+/B) to be explicitly included in the gen-
a user can be invited to directly select parts of a sentence
                                                                    eralized patterns, given that compound concepts triv-
and assigning them to meanings (e.g. “actor”, “claim”,
                                                                    ially correspond to ontological relationships of OIE inter-
“aggressor”, etc.), without having to see or interact with
                                                                    est, e.g.: “Film director David Lynch” implies that David
hypergraphic notation. Such refinements are beyond the
                                                                    Lynch is a film director.
scope of this work, but it is our hope to lay the founda-
                                                                       We considered the 50 most common such patterns,
tions for these and other possibilities.
                                                                    which we present in the appendix, in table 14. Following
                                                                    the process described in section 5.3 and the annotation
5.4 Open Information Extraction                                     guidelines document provided with the benchmark [35],
                                                                    we found that 36 of these patterns can be transformed
We will now show how 5 simple hyperedge patterns are
                                                                    into valid OIE relationships, given correct parses.
sufficient to rank first in a recent Open Information Ex-
                                                                       We then compressed these 36 patterns into the most
traction (OIE) benchmark [34]. In fact, one pattern is
                                                                    general ones that: (a) imply one or more of the original
even sufficient to surpass a majority of the systems of
                                                                    patterns, and (b) do not imply patterns found to be in-
that benchmark. We recognize the limitations of such
                                                                    correct in some way. For example, the two patterns:
benchmarks and do not claim that we have the best per-
forming OIE system, neither are we singly focused on                          (+/B.{ma} (ARG1/C...) (ARG2/C...))
this application. Instead, we are interested in providing                     (+/B.{mm} (ARG1/C...) (ARG2/C...))
empirical evidence for the expressive power of SH pat-
terns for the general purpose of knowledge extraction.              are compressed to:
   To discover OIE patterns, we took advantage of the                                (+/B.{m[ma]} */C */C)
Wikipedia part of the open text corpus that we developed
to train and validate the parser, and that we discussed in             Such a compression/generalization process could fea-
section 4 – Wikipedia content is a naturally rich source of         sibly be algorithmically automated.
factual assertions in NL. The resulting hypergraph con-                We thus arrived at the 5 patterns which are shown
tains 62528 top-level hyperedges.                                   in table 6. The extracted variables imply the usual OIE
   We then employed a simple process of generalization              tuples: 〈REL, ARG1, ARG2, ARG3...〉, with argument(s)
to transform hyperedges into abstract patterns. It con-             ARG3... being optional. Naturally, we convert hyper-
sists in replacing each element of a hyperedge with its             edges to the actual text they correspond to before feed-
corresponding type-annotated wildcard, for example:                 ing them to the benchmark. In the absence of REL, the
                                                                    relationship “is” is assumed. In some cases (patterns 3,
     (is/P.sc aragorn/C (of/B.ma king/C gondor/C))
                                                                    4, 5), notice also that REL is split into two or thee vari-
  2 https://github.com/graphbrain/graphbrain                        ables: REL1, REL2 and REL2. We just concatenate their


                                                               16
              #   Pattern                                                      Extractions   F1 (cumulative)   Rank
              1   (REL/P.{[sp][cora]x} ARG1/C ARG2 ARG3...)                        107            .265          4
              2   (+/B.{m[ma]} (ARG1/C...) (ARG2/C...))                             38            .311          3
              3   (REL1/P.{sx}-oc ARG1/C (REL2/T ARG2))                             20            .334          3
              4   (REL1/P.{px} ARG1/C (REL2/T ARG2))                                12            .351          2
              5   (REL1/P.{sc} ARG1/C (REL3/B REL2/C ARG2/C))                       16            .365          1

Table 6: Open Information Extraction patterns, ordered by decreasing contribution to F1 (presented cumulatively).
Ranks correspond to the rank achieved in the benchmark of Table 7 by using patterns up to the given line.


textual representation in the order indicated by the vari-           however this is not needed for our comparison with the
able names, with interleaving space characters.                      OIE benchmark.
                                                                        For a non-symmetrical example with (+/B.ma), let us
   Notice that the first pattern is almost a tautology of the        consider the sentence: “Finnish police reprimanded a
SH representation itself, producing triples where the first          man for traveling in a car boot to hide his meeting with
argument is the active or passive subject, the relation is           Prime Minister Juha Sipila during a government crisis
the predicate, and the second argument is the direct or              last summer, saying this was breach of the traffic code”,
indirect object, or complement, or agent, with the op-               with the emphasized concept parsed as:
tional argument being one of the specifications, if they
exist. To illustrate with a real and straightforward exam-           (+/B.am (+/B.am prime/C minister/C) (+/B.am juha/C
ple from the benchmark, consider the sentence: “The                                      sipila/C))
population of the special wards is over 9 million people,
with the total population of the prefecture exceeding 13             Using the same pattern, this leads to the single extrac-
million”. It is parsed to:                                           tion: 〈Juha Sipila, is, Prime Minister〉, while avoiding the
                                                                     potentially excessive generalization: 〈Prime Minister, is,
     (is/P.scx (of/B.ma (the/M population/C) (the/M                  Juha Sipila〉. We do know that Juha Sipila is one Prime
    (special/M wards/C))) ((over/M (9/M million/M))                  Minister, but not necessarily the only one in the context.
   people/C) (with/T (exceeding/P.so (of/B.ma (the/M                 The restriction requiring non-atomic edges in the argu-
  (total/M population/C)) (the/M prefecture/C)) (13/M                ments of this pattern is a simple mechanism to avoid too
                       million/C))))                                 trivial, and potentially silly inferences, such as “Barack is
                                                                     Obama”.
It matches pattern 1 with variables REL = is/P.scx;                     As a final example, let us illustrate pattern 3 with the
ARG1 = (of/B.ma (the/M population/C) (the/M (special/M               sentence: “Gonzales graduated from Crescent School in
wards/C))); ARG2 = ((over/M (9/M million/M)) people/C                Toronto, Ontario, Canada”, parsed as:
and ARG3 = (with/T (exceeding/P.so (of/B.ma (the/M (to-
                                                                      (graduated/P.sx gonzales/C (from/T (in/B.ma (+/B.am
tal/M population/C)) (the/M prefecture/C)) (13/M mil-
                                                                        crescent/C school/C) (,/J toronto/C (,/J ontario/C
lion/C))), resulting in the extraction: 〈the population of
                                                                                          canada/C)))))
the special wards, is, over 9 million people, with the total
population of the prefecture exceeding 13 million〉.                  and resulting in extractions where the first part of
   Pattern 2 can also be seen as a direct consequence of             REL is extracted from the predicate and the second
SH representation, in this case inferring ontological rela-          from the trigger: 〈Gonzales, graduated from, Crescent
tionship from the (+/B) builder structure. Cases where               School in Toronto〉. The previously discussed conjunc-
both arguments have the role “m” can be interpreted as               tion decomposition process leads to the further extrac-
two expressions of the same concept. Again, using a real             tions: 〈Gonzales, graduated from, Crescent School in
example, the emphasized part of the sentence “He is the              Ontario〉 and 〈Gonzales, graduated from, Crescent School
younger brother of the prolific film composer Christophe             in Canada〉.
Beck” was parsed as:                                                    We stop illustrating how these patterns apply here for
                                                                     the sake of succinctness, but hope to have sufficiently
      (+/B.mm (the/M (prolific/M (+/B.am film/C                      shown how they are generic and straightforward manip-
     composer/C))) (+/B.am christophe/C beck/C))                     ulations of the structures enabled by SH.
leading to the symmetrical extractions: 〈the prolific film              Table 7 shows the full benchmark, comparing the per-
composer, is, Christophe Beck〉 and 〈Christophe Beck, is,             formance of our approach with seven other methods.
the prolific film composer〉. We discuss below in sec-                SH outperformed all baseline systems in 24.6% of the
tion 6.4 how easy it is to further extract, from this point          cases that tend to consist of complicated combinations
and thanks to the recursive hypergraphic structure, the              of conjunctions and prepositional phrases. For exam-
relation 〈Christophe Beck, is, film composer〉; for now               ple, the sentence where the next best system is defeated


                                                                17
     System                              Extractions   Matches      Exact     Prec. of   Recall of   Prec.   Recall   F1
                                                                   matches    matches    matches
     Semantic hypergraphs with 5 rules      201         120           19        .70        .93       .416    .326     .365
     MinIE [26]                             252         134           10        .75        .83       .400    .323     .358
     ClausIE [20]                           223         121           24        .74        .84       .401    .298     .342
     OpenIE 4 [41]                          101          74            5        .68        .84       .501    .182     .267
     Semantic hypergraphs with 1 rule       107          74            7        .69        .85       .475    .184     .265
     Ollie [40]                             145          74            8        .73        .81       .347    .175     .239
     ReVerb [25]                             79          54           13        .83        .77       .569    .121     .200
     Stanford [4]                           371          99            2        .79        .65       .210    .188     .198
     PropS [71]                             184          69            0        .59        .80       .222    .162     .187

Table 7: Performance of OpenIE systems, ordered by descending F1. Bold figures indicate the best performing system
for each category.


by the highest margin is: “A very detailed treatment of          For example, we know that (germany/C) is related in an
the EM method for exponential families was published             unspecified way to (of/B.ma capital/C germany/C). Of
by Rolf Sundberg in his thesis and several papers fol-           course, this is not to say that a more specific relation can-
lowing his collaboration with Per Martin-Löf and Anders          not be inferred by further processing with other meth-
Martin-Löf.” Furthermore, the mean number of words               ods. Here we are simply highlighting the ontological re-
per sentence where SH is not the best is 21.2, vs. 23.8          lations that come “for free” with the hypergraphic repre-
where it is the best (31.0 when outperforming by a fac-          sentation.
tor ≥ 1.5), plausibly indicating an advantage with more             When parsing sentences to hyperedges, and taking
complicated sentences.                                           advantage of another classical NLP task offered by the
                                                                 upstream package, we also store auxiliary hyperedges
                                                                 connecting every atom that corresponds to word to the
6 Computations on the hypergraph:                                lemma of that word, with the help of a special connector
                                                                 “lemma/J”. For instance:
  concepts, ontologies and corefer-
                                                                                 (lemma/J saying/P say/P)
  ence resolution
                                                                    In the next section we will make use of this, but it is
We have shown how SH representation makes it possi-              easy to see how lemmas facilitate the inference of various
ble to infer knowledge using simple symbolic rules. We           types of correspondences, for example between singular
will now address how it enables knowledge inference              and plural forms such as (apple/C) and (apples/C) with
using probabilistic and heuristic rules. More specifi-           the help of (lemma/J apple/C apples/C), and thus more
cally, we will show how to derive ontologies and per-            sophisticated structural variations such as (+/B.am ap-
form coreference resolution among concepts. For exam-            ple/C season/C) and (of/B.ma season/C apples/C).
ple, how automated methods can reach the conclusion
that “President Obama” is a type of “President”, or that
“Obama”, “Barack Obama” and “President Obama” re-                6.2 Coreference resolution: co-occurrence
fer to the same external entity, while “Michelle Obama”              graph
refers to another one. Before proceeding, let us consider
                                                                 A common but challenging task in NLP is that of
implicit taxonomies.
                                                                 coreference resolution, a usual disambiguation issue
                                                                 which consists in identifying different sequences of n-
6.1 More about concepts and implicit tax-                        grams that refer to the same entity (such as “Barack
    onomies                                                      Obama” and “President Obama”). This is an old research
                                                                 topic [65] that has been revived lately with modern ma-
Hyponyms of a concept can be found by looking for hy-            chine learning methods [55]. ML approaches such as
peredges where the concept appears either as the main            deep learning require large training sets and tend to pro-
argument of a builder-defined concept or as the argu-            vide black box models, where precision/recall can be
ment of a modifier-defined concept. It follows from              measured and improved upon, but the exact mecha-
these structures that the SH representation implicitly           nisms by which the models operate remain opaque. Here
builds a taxonomy. More generally, we can talk of an             we do not mean to provide a complete solution to this
implicit ontology. Beyond the taxonomical relationships          problem, but instead show that several cases of corefer-
that we described, the concepts that form a concept              ence resolution can be performed in a simpler and un-
hyperedge are related to it in a non-specified fashion.          derstandable manner through the use of semantic hy-


                                                            18
Figure 4: Example of coreference resolution. On the left panel we can see the co-occurrence graph and its compo-
nents, identified by different colors and leading to corresponding coreference sets on the right panel. The probabili-
ties for each coreference set are shown to their left, including the ratios of total degree of the set to total degree, used
to compute them. Individual degrees are shown next to each edge. * indicates the assignment of the seed to one of
the coreference sets. ** indicates the recursive nature of the process, with (+/B michelle/C obama/C) taking the role
of seed in another instance of this coreference resolution process.


pergraphs for situations that are nevertheless common               of all the auxiliary concepts that appear together with
and useful, especially in the context of social science re-         the seed. A connection between two concepts in this
search.                                                             graph means that there is at least a compound concept
   We will discuss in the following section several exper-          in which they take part together. In the example, we can
imental results that we obtained on a dataset of several            see that this graph has three maximal cliques, which we
years of news headlines. This corpus is largely focused             identified with different colors. We then apply this sim-
on political issues, and it is dominated by reports of ac-          ple rule: two compound concepts are assumed to refer to
tors of various types making claims or interacting with             the same entity if all of their auxiliary concepts belong to
each other. These actors can be people, institutions,               the same maximal clique. The intuition is that, if auxil-
countries and so on. In our hypergraphic representa-                iary concepts belong to a maximal clique, then they tend
tion, such actors will very frequently be referred to by hy-        to be used interchangeably along with the seed, which
peredges forming compound nouns, with the use of the                indicates that they are very likely to refer to the same en-
(+/B) connector, as discussed previously.                           tity. We will show that this intuition is empirically con-
   In figure 4 we can see such a case: a number of                  firmed in our corpus, from where the example in the fig-
compound concept edges with the main atomic con-                    ure was extracted.
cept (obama/C) refer to actors. How can we group them
in sets, such that all the cases in a given set refer to
                                                                       The co-occurrence graph method produces the coref-
the same entity? Here, we start taking advantage of the
                                                                    erence sets seen on the right of the figure, except for
hypergraph as a type of network, and of the analysis
                                                                    the items marked with * and **. As can be seen, it cor-
graphs that we can easily distill from the hypergraph.
                                                                    rectly groups several variations of hyperedges that refer
Semantic graph-based disambiguation has been exten-
                                                                    to Barack Obama (president of the United States dur-
sively explored since the mid-2000s, especially empha-
                                                                    ing most of the time period covered by our news cor-
sizing the importance of centrality and proximity in de-
                                                                    pus), and it correctly identifies a separate set referring
ciding which sense correspond to a given word in a cer-
                                                                    to Michelle Obama, his wife. It can also be seen that it
tain context, and semantic hypergraphs are no exception
                                                                    fails to identify that “Mr. Obama” is also likely to refer to
[43, 51, 2].
                                                                    Barack Obama. We will say more about this specific case
   We can trivially traverse all the concepts in the hyper-
                                                                    when we discuss claim analysis, in the next section.
graph, finding the subset of concepts that play the role of
main concept in the above mentioned compound con-
cept constructs. For each of these seed concepts, we can               But what about the seed concept itself, in this case
then attempt to find coreference relationships between              (obama/C)? The co-occurrence method is not able to as-
the concepts they help build. In the figure, we see an ex-          sign it to one of the sets. Here we employ another simple
ample using the seed concept (obama/C). On the right                method, this time of a more probabilistic nature. Before
side of the figure, we see all the compound concepts con-           tackling this method, we have to make a small detour to
taining the seed as the main concept (except for the ones           discuss the semantic hypergraph from a network analy-
marked with * and **). It is possible then to build a graph         sis perspective.


                                                               19
6.3 Simple hypergraph metrics                                        germany/C)) also have deep degree δ = 1, but the lat-
                                                                     ter (germany/C) has deep degree δ = 2, because not only
In a conventional graph, it is common to talk of the de-
                                                                     does it participate directly in the edge (of/B capital/C ger-
gree of a vertex. This refers simply to the number of
                                                                     many/C), but it also participates at a deeper level in the
edges that include this vertex or, in other words, the
                                                                     outer edge (is/P berlin/C (of/B capital/C germany/C)). In
number of other vertices that it is directly connected
                                                                     other words, the higher deep degree of (germany/C) indi-
to (we assume here an undirected graph without self-
                                                                     cates that it plays an increased role as a building block in
loops). With a semantic hypergraph, such measure is
                                                                     other edges.
not so straightforward, given that an edge can have more
than two participants, and that recursivity is permitted.
   Let us first define the set D e , containing all the edges        6.4 Coreference resolution:                probabilistic
in with a given edge e participates:                                     seed assignment
                  D e = {e i |e i ∈ E ∧ e ∈ e i }         (1)        Back to figure 4, each coreference set is labeled with a
                                                                     probability p, representing the chance that a given seed
We define the degree of a hyperedge e as:                            appears in one of its edges, if we were to uniformly enu-
                          X ¡            ¢                           merate all edges that rely on this seed. This configures
                 d (e) =      |e i | − 1                  (2)        a simple estimation of the probability of the seed by it-
                            e i ∈D e
                                                                     self being used with a certain meaning, represented by
   This is to say, the hypergraphic degree is the number             the given coreference set. These probabilities are thus
of edges with which a given edge is connected to by outer            the ratio between the sum of the degrees of the edges in
hyperedges. It is intuitively equivalent to the conven-              each coreference set and the total degree of all edges that
tional graph degree.                                                 include the seed, i.e. of all coreference sets.
   Another useful metric that we can define is the deep                  Two simple heuristics drive this step. One is that peo-
degree, which considers edges connected by hyperedges                ple will tend to use an ambiguous abbreviation of a con-
not necessarily at the same level, but appearing recur-              cept when the popularity of one of the interpretations is
sively at any level of the connecting hyperedge. Let                 sufficiently high in relation to all the others. For exam-
us consider the set ∆e , containing the edges that co-               ple, both (+/B barack/C obama/C) and (+/B michelle/C
participate in other edges with e at any level. This set             obama/C) share the seed (obama/C), but when referring
is recursively defined, so we describe how to generate it,           only to (obama/C) during the period he was a US pres-
in Algorithm 2.                                                      ident, people tend to assume that it refers to the most
                                                                     frequently mentioned entity – Barack Obama. The other
                                                                     is that a given seed should only be considered as an ab-
  Function Generate∆(e)                                              breviation if it is used a sufficient amount of times as a
     Data: An edge e
                                                                     primary concept in relations, i.e. if there is evidence that
     Result: ∆e neighborhood of edge e
                                                                     it is in fact used on its own to refer directly to some ex-
     ∆e ←− D e
                                                                     ternal concept, and not only as a common component
     for e 0 ∈ D e do
                                                                     of primary concepts. Put differently, seeds referring to
        ∆0 ←− Generate∆(e’ )
                                                                     common concepts which act often as building blocks of
        ∆e ←− ∆e ∪ ∆0
                                                                     other concepts (i.e., higher deep degree with respect to
     end for                                                         degree) are less likely to be valid abbreviations. Such is
     return ∆e                                                       the case for “house” (which may indifferently refer to the
  end                                                                White House or Dr. House) and “qaida” (which is typi-
 Algorithm 2: Generating the neighborhood ∆e of                      cally used as a building block for Al Qaida and never by
 an edge e.                                                          itself ).
                                                                         We thus establish a criterion that consists of the ful-
We can now define the deep degree δ as:                              fillment of each of these two conditions, corresponding
                                                                     respectively to the heuristics above. A given seed s is as-
                 δ(e) =
                        X ¡             ¢
                             |e i | − 1                   (3)        signed to the coreference set C with the highest p if:
                            e i ∈∆e
                                                                       • p is above a certain threshold θ
   To provide a more intuitive understanding of these
metrics, let us consider the edge “(is/P berlin/C (of/B                • d s /δs is above a certain threshold θ 0
capital/C germany/C))”. Let us also assume that no other
edges exist in the hypergraph. In this case, the edges                 We set the threshold to the values θ = .7 and θ 0 = .05,
(is/P), (of/B capital/C germany/C) and (germany/C) all               that we verified empirically to produce good results. Nat-
have degree d = 1, because they all participate exactly              urally, these thresholds can be fine-tuned using methods
in one edge. The first two ((is/P) and (of/B capital/C               more akin to hyper-parameter optimization in ML, but


                                                                20
such optimizations are outside of the scope of this work.           fully integrated such a system3 with the Graphbrain li-
When the criterion is not met, the seed is left as a refer-         brary, but avoid using it in this work for the sake of sim-
ence to a distinct entity. In our corpus, this happens for          plicity while defining SH methodological foundations.
example with "Korea", which remains an ambiguous ref-
erence to either “North Korea” or “South Korea”.
                                                                    7 Integrated Case Study: Claim and
                                                                      Conflict Analysis
6.5 Further disambiguation cases
                                                                    We arrive at the point where we can propose an inte-
We do not present here a general solution for corefer-              grated application of the formalisms and methods dis-
ence resolution and synonym detection, let alone disam-             cussed so far to the analysis of a large corpus of real
biguation as a whole. Some further cases beyond coref-              text, combining symbolic and probabilistic rules. More
erence resolution will nonetheless be covered in the next           specifically, we worked with a corpus of news titles that
section, notably anaphora resolution, given that this re-           were shared on the social news aggregator Reddit. We
quires the discussion of predicates and relations in more           extracted all titles shared between January 1st, 2013 and
detail, and along with empirical results. Other cases are           August 1st, 2017 on r/worldnews, a community that is
left out of this work, but we would like to provide a quick         described as: “A place for major news from around the
insight into how they may be treated.                               world, excluding US-internal news.” This resulted in a
   One obvious example is that of synonyms, which are               corpus of 404,043 news titles. We applied the methods
not implied by a pure structural analysis of hyperedges             described in sections 4 and 6 to generate a hypergraph
– e.g. red and crimson, as well as U.S. and United States,          from the titles.
for they share no common seed (as opposed to the cases                 We decided to focus on two specific categories of ut-
emphasized in the previous subsection). This type of                terances that are very frequent in news sources, and of
synonym detection may be achieved with the help of a                special interest for the social sciences [72], especially the
general-knowledge ontology such as Wordnet or DBPe-                 study of public spaces [59, 74]: a claim made by an ac-
dia, and/or with the help of word embeddings such as                tor about some topic and an expression of conflict of one
word2vec. This is a foreseeable and desirable improve-              actor against another, over some topic. Helpfully, the de-
ment to hypergraph-based text analysis that we leave for            tection of such categories also allows us to illustrate sim-
future work.                                                        ple symbolic inference over the hypergraph.
   Another case is the inverse problem of synonym
detection: disambiguating atoms that correspond to
the same word but to different entities, for exam-                  7.1 Knowledge Inference
ple distinguishing “Cambridge (UK)” from “Cambridge                 The English language allows for vast numbers of verb
(USA/Massachusetts)”. We do not perform this type of                constructions that indicate claims or expressions of con-
distinction in this work, but we present another syntac-            flict. Instead of attempting to identify all of them, we
tic detail that enables them from a knowledge represen-             considered the 100 most common predicate lemmas in
tation perspective: the atom namespace. Quite sim-                  the hypergraph, and from there we identified a set of
ply, beyond the human-readable part and the type and                “claim predicates” and a set of “conflict predicates”, that
other machine-oriented codes, a third optional slash-               we detail below. Overall, we found 3730 different predi-
separated part can be added to atoms, allowing to dis-              cate lemmas, and their rank-frequency distribution is ex-
tinguish them in cases such as the above, e.g.: cam-                pectedly heavy-tailed. In this case, this small fraction of
bridge/C/1 and cambridge/C/2.                                       the set accounts for 60.6% of the hyperedges. Naturally,
   Finally, coreference resolution can also apply to cases          coverage could be improved by considering more predi-
where neither seed concepts are shared, nor anaphoras               cates, but with diminishing returns.
are present. Let us say that one sentence refers to                    We then employed the same process described in sub-
“Kubrick” and the next one to “the film director”. Both             section 5.3 to discover rules that are capable of detecting
this type of case and the above mentioned disambigua-               claims and expressions of conflict, whereby a hyperedge
tion cases are likely to be more easily solved with the help        contains an attributable:
of structured knowledge surrounding the concepts in
the semantic hypergraph, eventually including general                 • claim, if the following conjunction of patterns is sat-
knowledge as mentioned. For example, it could be de-                    isfied:
tected that a certain reference to “Cambridge” is closer to
references related to the United States, or that “Kubrick”                  (PRED/P.{sr} ACTOR/C CLAIM/[RS])
is structurally close to the concept of “film director”. Al-
                                                                                     ∧ (lemma/J >PRED/P [say,claim]/P)
ternatively, a hybrid approach taking advantage of deep
learning models can be employed. In fact, we success-                 3 https://github.com/huggingface/neuralcoref




                                                               21
(a) “North Korea says it's not afraid of US military strike”                   (b) “Germany warns Russia against military engagement in Syria”
       (says/P.sr      outer predicate                                                  (warns/P.sox     predicate

           (+/B.am north/C korea/C)                   outer subject                         germany/C      subject

           ((not/M ’s/P.sc)           inner predicate                                       russia/C     object

               it/C     inner subject                                                       (against/T          trigger
 relative
 relation      (of/B.ma                                                                         (military/M                                    specification
                    afraid/C                                             inner                       (in/B.ma engagement/C syria/C))))
                                                                      complement
                    (military/M (+/B.am us/C strike/C)))))



                    Figure 5: Two examples of relations starting with either a claim or a conflict predicate.


   • expression of conflict, if the following conjunction                                   task                 error type                error      error
                                                                                                                                           count       rate
     of patterns is satisfied:
                                                                                            claim inference      not a claim               0/100       0%
                                                                                                                 wrong actor               0/100       0%
         ( PRED/P.{so,x} SOURCE/C TARGET/C                                                                       wrong topic               2/100       2%
                                                                                                                 bad anaphora resolution    1/13       8%
                 [against,for,of,over]/T TOPIC/[RS] )                                                            minor defects in topic    8/100       8%
                         ∧ ( lemma/J >PRED/P                                                conflict inference   not a conflict            0/100       0%
            [accuse,arrest,clash,condemn,kill,slam,warn]/P )                                                     wrong origin actor        0/100       0%
                                                                                                                 wrong target actor        0/100       0%
                                                                                                                 wrong topic               0/100       0%
                                                                                                                 minor defects in topic    4/100       4%
So, a claim is essentially a relation, based on a predi-
cate of lemma “say” or “claim”, between an actor and a
                                                                                        Table 8: Evaluation of several types of error in claim and
claim, which may also be a relation or a specifier. These
                                                                                        conflict inference. Error counts and rates are presented,
patterns additionally rely on a new notation, “>”. As we
                                                                                        based on the manual inspection of 100 randomly se-
have seen in section 3, a predicate can be a non-atomic
                                                                                        lected claims and 100 randomly selected conflicts.
hyperedge. As with concepts, the meaning of predicate
atoms can be extended with a modifier. For example, the
English verb conjugation “was saying” is represented as                                 (claims/P.sr (+/B.am google/C boss/C) ((does/M (not/M
(was/M saying/P). Eventually, there is always a predicate                                 know/P.so)) he/C (in/B.ma (his/M salary/C) (+/B.am
atom that corresponds to the main verb in the predicate:                                                commons/C grilling/C))))
the notation refers to the innermost atom, i.e. removing
an arbitrary amount of nesting based on modifiers. For                                  In this case, the concept “commons grilling” should be a
example, “>PRED” matches “been/P”, “(has/M been/P)”,                                    separate specification:
“(not/M (has/M been/P))”, and so on.
                                                                                        (claims/P.sr (+/B.am google/C boss/C) ((does/M (not/M
                                                                                           know/P.sox)) he/C (his/M salary/C) (in/T (+/B.am
   In figure 5 we present two real sentences from our cor-
                                                                                                        commons/C grilling/C))))
pus and their respective hyperedges. Example (a) was
classified as a claim and example (b) as an expression of
conflict. These examples were purposely chosen to be
                                                                                        Subjects and actors. Both claim and conflict structures
simple, but the above rules can match more complicated
                                                                                        imply that the hyperedge playing the role of subject in
cases. For example, the following sentence was correctly
                                                                                        the relation is an actor. Using the methods described
identified and parsed as a claim:
                                                                                        in section 6, we can identify the coreference set of each
                                                                                        actor and replace all occurrences of this actor with the
      U.S. Secretary of State John Kerry was the in-
                                                                                        same hyperedge. For each coreference set we choose
      tended target of rocket strikes in Afghanistan’s
                                                                                        the hyperedge with the highest degree as the main iden-
      capital Saturday, the Taliban said in a state-
                                                                                        tifier, following the heuristic that the most commonly
      ment claiming responsibility for the attacks.
                                                                                        used designation of an entity should be both recogniz-
                                                                                        able and sufficiently compact.
Validation. In table 8 we present an evaluation of accu-                                   As seen in figure 5(a), the inner subject (i.e., the sub-
racy based on the manual inspection of 100 claims and                                   ject of the relative relation that represents what is being
100 conflicts that were randomly selected from the hy-                                  claimed) can be a pronoun. These cases are very com-
pergraph. Defects are deemed to be minor if they do not                                 mon, and almost always correspond to a case where the
interfere with the overall meaning of the hyperedge (e.g.,                              actor is referencing itself in the content of the claim. On
by leading to one of the other, more serious errors listed                              one hand, we perform simple anaphora resolution: if the
in the table). To illustrate with a minor defect from our                               inner subject is a pronoun in the set {he/C, it/C, she/C,
dataset:                                                                                they/C}, then we replace it with the outer subject. On the


                                                                                   22
    Type      Rank    Actor              Hyperedges (coreference set)                                                       Degree
                  1   China              china/C, (+/B.am south/C china/C)                                                    6199
 non-human        2   Russia             russia/C                                                                             5861
                  3   U.S.               us/C, (the/M us/C)                                                                   3824
                  8   Vladimir Putin     (+/B.am president/C putin/C), putin/C, (+/B.am vladimir/C putin/C), (+/B.am          2338
                                         president/C (+/B.am vladimir/C putin/C)), (+/B.am (russian/M president/C)
                                         (+/B.am vladimir/C putin/C)), (+/B.am (russian/M president/C) putin/C)
                 10   Barack Obama       (+/B.am (+/B.am us/C president/C) (+/B.am barack/C obama/C)), (+/B.am                2069
                                         president/C obama/C), (+/B.am barack/C obama/C), (+/B.am president/C
    male
                                         (+/B.am barack/C obama/C)), obama/C, (+/B.am (+/B.am u.s./C president/C)
                                         (+/B.am barack/C obama/C))
                 23   Donald Trump       (+/B.am president/C (+/B.am donald/C trump/C)), (+/B.am (+/B.am us/C pres-           1082
                                         ident/C) (+/B.am donald/C trump/C)), (+/B.am donald/C trump/C), trump/C,
                                         (+/B.am president/C trump/C)
                 32   Angela Merkel      merkel/C, (+/B.am angela/C merkel/C), (+/B.am (german/M chancellor/C)                 750
                                         (+/B.am angela/C merkel/C)), (+/B.am chancellor/C (+/B.am angela/C
                                         merkel/C)), (+/B.am (german/M chancellor/C) merkel/C)
   female
                 78   Theresa May        may/C, (+/B.am theresa/C may/C), (+/B.am (+/B.am prime/C minister/C)                  270
                                         (+/B.am theresa/C may/C))
               201    Nicola Sturgeon    sturgeon/C, (+/B.am nicola/C sturgeon/C)                                               81
                46    The Palestinians   palestinians/C, (the/M palestinians/C)                                                487
   group        70    The Kurds          kurds/C, (the/M kurds/C)                                                              302
               113    The Russians       russians/C, (the/M russians/C)                                                        184


Table 9: Three actors with highest hypergraphic degree in each category: non-human, male, female, group (in de-
creasing order of highest degree).


other hand, we take advantage of the pronoun to infer                   generated by it. Human observers can then infer some
more things about the actor. The four pronouns men-                     higher-level concept from these words and probabilities.
tioned indicate, respectively, that the actor is a male hu-             For example, if the five highest probability words for a
man, a non-human entity, a female human, or a group.                    topic X are {EU, Ursula von der Leyen, Boris Johnson,
We take the majority case, when available, to assign one                Barnier, Trade}, a human observer may guess that a good
of these categories to actors. The pronoun they is being                label for this topic is Brexit Negotiations. LDA is appli-
increasingly used as a gender-neutral third person singu-               cable to sets of documents, for a predefined number of
lar case, but we have not found such cases in our corpus.               topics, where each document is considered to be a bag-
   Table 9 shows the top three actors per category, ranked              of-words. Distributional topic detection methods have
by their degree in the hypergraph, along with their coref-              recently generated a variety of research endeavors, in-
erence set. Obviously, more sophisticated rules can be                  cluding the application of stochastic blockmodels to dis-
devised, both for anaphora resolution and category clas-                cover joint groups of documents which use keywords in
sification. Our goal here is to illustrate that, thanks to the          a similar fashion [27].
SH abstraction, it becomes possible to perform powerful                    A different approach to topic detection is Tex-
inferences (i.e., both useful and at a high level of seman-             tRank [44], which is capable of detecting topics within a
tic abstraction) with very simple rules.                                single document. With TextRank, the document is first
                                                                        transformed into a word co-occurrence graph. Com-
                                                                        mon NLP approaches are used to filter out certain classes
7.2 Topic Structure
                                                                        words from the graph (e.g., do not consider articles such
The very definition of topic, for the purpose of automatic              as “the”). Topics are considered to be the words with
text analysis, is somewhat contingent on the method be-                 the highest network centrality in this graph, according
ing employed. One of the most popular topic detection                   to some predefined threshold condition. Simple statis-
methods in use nowadays is Latent Dirichlet Allocation                  tical methods over the co-occurrence graph can be used
(LDA) [12], which is a probabilistic topic model [11] that              to derive ngram topics from the previous step. Given that
views topics as latent entities that explain similarities be-           the order in which words appear in the document is im-
tween sets of documents. In LDA, topics are abstract                    portant, TextRank cannot be said to be a bag-of-words
constructs. Documents are seen as a random mixture of                   approach such as LDA. It relies a bit more on the mean-
topics, and topics are characterized by their probability               ing of the text, and it is more local – in the sense that it
of generating each of the words found in the document                   works inside a single document instead of requiring sta-
set. LDA uses a generative process to statistically infer               tistical analysis over a corpus of documents.
these probabilities. Ultimately, a topic is described by                   In this work, we move significantly more in the direc-
the set of words with the highest probabilities of being                tion of text understanding and locality. Our topics are


                                                                  23
      actor                            topic
                                   →   scuppering syria peace talks
 =⇒   assad
                                   →   war crimes in aleppo
 =⇒   damascus                     →   continuing to use chemical weapons
 =⇒   the united states            →   the weakening of europe
 =⇒   us                           →   espionage
                               ←       mistral delay
 ⇐⇒   russia                       →   meddling in election
                                   →   rapid eu sanctions
                               ←       being europe’s biggest problem child
 ⇐⇒   germany
                                   →   wage dumping in the meat sector
 ⇐=   al qaeda                 ←       new attacks
 ⇐=   council of europe        ←       allowing to hit parents and spank their children
 ⇐=   iraqis                   ←       imminent isis attacks
 ⇐=   kagame                   ←       rwanda genocide
 ⇐=   london                   ←       plot
 ⇐=   syria’s assad            ←       supporting terrorism
 ⇐=   un                       ←       racist attacks on black minister
 ⇐=   united nations top hu-   ←       delays
      man rights official


Table 10: List of actors criticizing or being criticized by ego (here, France), and the topics over which the critique ap-
plies. Single arrows show the critique direction (left to right: ego criticizes that actor) for each underlying hyperedge,
double arrows indicate the overall critique direction (which can thus go both ways).


firstly inferred from the meaning of sentences. As we                           These chunks are then used as textual labels for the hy-
have shown, pattern analysis of hyperedges can be used                          peredges.
to infer relationships such as claim and conflict, which
imply both actors and topics. Given coreference detec-
tion, such topics are characterized by sets of hyperedges,                      7.3 Inter-actor criticism
but these sets are not probabilistic in the sense that LDA’s                    Focusing on France and Germany as target actors a, we
are. Instead, they are a best guess of symbolic represen-                       gather the results for the detection of conflict patterns in
tations that map to some unique concept. Our approach                           the tables 10 and 11. Each of these actors is involved in
relies even more on meaning than TextRank, and it al-                           active or passive criticism of other actors, i.e. either crit-
lows for topic detection at an even more local scale: sin-                      ical (→ ) of or criticized by (←) other actors. The critique
gle sentences.                                                                  is related to a topic, and may go in both directions, i.e.
   In the examples given in figure 5, the claim shown in                        Germany criticizes Greece for debt commitments (sec-
(a) implies the rather specific topic “afraid of military us                    ond row of table 11).
strike”, and (b) the topic “military engagement in syria”.                         The topics presented here correspond to the detailed
   Another important aspect of our approach is that top-                        topics discussed in the previous section. This structured
ics can be composed of other topics or concepts, forming                        enumeration provides a way to scan the direction, target
a hierarchical structure. This is a natural consequence                         and frequency of claims by actors on other actors in a
of how we model language, as explained in section 6.1.                          given text corpus.
This allows us to explore topics at different levels of de-
tail. The topic implied by a claim or conflict can be very
specific and possibly unique in the dataset, but the more
                                                                                7.4 Dyadic claims
general subtopics or concepts that it contains can be                           Here we focus on claims that actors make about other
used to find commonalities across the hypergraph. Con-                          actors (or themselves). In other words, we refer to claims
sidering the hyperedge from one of the topic examples                           where the subject of the claim is itself an actor. Fur-
above, from (military/M (in/B engagement/C syria/C)) it                         thermore, we consider only claims for which the claim
is possible to extract concepts from inner edges that cor-                      relation contains an argument playing the rule of com-
respond to more general concepts, for example (syria/C),                        plement, meaning that the subject of the claim is being
(engagement/C) and (in/B engagement/C syria/C). With                            linked with some concept, for example expressing mem-
the help of the implicit taxonomy, which indicates that                         bership in a class (e.g.: “Pablo is a cat.”) or the possession
(in/B engagement/C syria/C) is a type of engagement/C, a                        of some property (e.g.: “North Korea is afraid”).
simple rule could also infer that (military/M (in/B engage-                        We also recursively extract context edges that are con-
ment/C syria/C)) is a type of (in/B engagement/C syria/C).                      nected to the outer claim edge through nestings of (:/J).
  In the various tables of results that we will subse-                          To give an example from our corpus:
quently present, actors and topics are represented by la-
bels in natural language. During the transformation of                            (:/J (says/P.sr russia/C (’s/P.sc it/C ready/C)) ((to/M
text to hyperedge, every hyperedge that is generated is                               deal/P.x) (with/T (new/M (+/B.am ukraine/C
associated with the chunks of text from which it comes.                                                president/C)))))


                                                                           24
      actor                        topic
 =⇒   fiat                     →   using illegal emissions device
 =⇒   greece                   →   debt commitments
 =⇒   israel                   →   latest settlement expansion in east jerusalem
 =⇒   kurds                    →   one sided referendum plans
 =⇒   maduro                   →   holding venezuelans’ hostage
 =⇒   mexico city              →   brexit
                               →   cold war reflexes
 =⇒   putin
                               →   moscow up beefs nuclear arsenal
 =⇒   syrian                   →   alleged car bomb plot
 =⇒   uk                       →   leaving eu
 =⇒   ukraine                  →   graft
 =⇒   us                       →   stasi methods ahead of obama
                           ←       halting arms deal
                               →   cyber attack on ukraine peace monitors
 ⇐⇒   russia
                               →   kremlin dismisses us intelligence claims as a witch hunt
                               →   military engagement in syria
                           ←       wage dumping in the meat sector
 ⇐⇒   france
                               →   being europe’s biggest problem child
                           ←       causing instability
 ⇐⇒   u.s.
                               →   ceding lead role to china
                           ←       backing failed coup
                           ←       cultural racism over eu accession
                           ←       engaging in diplomatic rudeness and double standards
                           ←       genocide speech
                           ←       harbouring terrorists
                           ←       succor providing to its enemies
 ⇐⇒   turkey               ←       succour providing to its enemies
                           ←       working against erdogan
                               →   blackmailing eu
                               →   itself further distancing from europe by the death penalty reinstating after a disputed referendum
                               →   monday
                               →   nazi
                               →   supporting terrorism
 ⇐=   erdogan              ←       nazi practices over blocked political rallies
 ⇐=   eu commission        ←       air pollution breaches
 ⇐=   eu leaders           ←       pressure on migrant quotas
 ⇐=   french far right     ←       doors opening to refugees
      leader marine le
      pen
 ⇐=   italy                ←       undermining its economic efforts
 ⇐=   moscow               ←       up hushing russian girl’s rape
 ⇐=   orban                ←       rude tone over refugees
 ⇐=   snowden              ←       nsa aiding in spying efforts
 ⇐=   turkey’s president   ←       behaving like nazis
      tayyip erdogan
 ⇐=   un                   ←       institutional racism and racist stereotyping against people of african descent
 ⇐=   un committee         ←       an anti racism - convention violating by not prosecuting a politician’s comments about turks and arabs


Table 11: List of actors criticizing or being criticized by ego (here, Germany), and the topics over which the critique
applies. Single arrows show the critique direction (left to right: ego criticizes that actor) for each underlying hyper-
edge, double arrows indicate the overall critique direction (which can thus go both ways).


   The edge “((to/M deal/P.x) (with/T (new/M (+/B.am                       past.
ukraine/C president/C))))” is extracted as a context edge.
Finally, specification edges of the claim and context                    • The presence of the modifier will/M implies the fu-
edges are extracted out and grouped together.                              ture tense.
   The predicate of the relative relation that expressed
the claim is inspected to further determine the tense                    In table 12, we present such attributions between the
of the attribution (present, past, future), and to identify            actors: North Korea, Russia, Putin and U.S.
negations. Once again, this is achieved by simple rules
over the hypergraphic representation:                                  7.5 Topic-based conflict network
  • The presence of a negation modifier (not/M, n’t/M),
                                                                       So far we have presented actor-centric results. Here we
    as is in fact the case with the first example of figure 5.
                                                                       will consider all conflicts that contain “Syria” as topic
  • The presence of the predicate was/P implies the                    or subtopic (according to the definitions of section 7.2).


                                                                  25
   source               target                property, context and <specification>
                                                    just the place for you
                                                    the victim of intensive cyberattacks
                                                    ready; to strike u.s. aircraft carrier
                                                    able; to nuke u.s. mainland
                                                    a great place for human rights
                                                    ready for war with us
                                                    close; developing a new satellite; speculation fuelling it might attempt a long range rocket to fire to mark a key
 north korea   says   north korea     is
                                                    political anniversary; <next month>
                                                    open; holding talks with south korea; <including the suspension of the south’s joint military drills with the
                                                    united states>; <if are met certain conditions>
                                                    open; to talk with south korea
                                                    the biggest victim in u.s. student’s death
                                                    responsible for righteous sony hacking
                                              not
                                                    afraid of us military strike

                                     was            ready; to put russia’s nuclear weapons; <during tensions over the crisis in ukraine and crimea>; <on standby>
                                                    possible convinced solution to ukraine crisis
                                                    willing; to play a mediating role between the two koreas; relieve the state of crisis on the korean peninsula;
                                                    <according to president moon jae in’s special envoy to moscow>; <to help>; <by dispatching an emissary>; <to
   putin                              is
                                                    pyongyang>
                                                    ready; to sell s-400 anti aircraft system; <to turkey>
                                              not   russia’s president for life
                                                    russia’s president; 2024
                                    will be
                                              not   president for life
               says     putin
                                                    ready; to improve ties with the us
                                                    moral compass of the world
                                                    interested; other brics brazil, russia, india, china, south africa members; using national currencies; <after
   russia                             is
                                                    agreeing on such an arrangement with china>
                                                    willing; over to hand to us house of representatives and senate
                                              not   a threat to anyone
                                     was            the mastermind of the ukrainian coup
     us
                                      is            world’s only superpower; back walks trump compliments

                                                    open; coordinating with u.s. in syria
                                                    ready; to deal with new ukraine president; to retaliate for u.s. election sanctions
   russia      says     russia        is            ready for dialogue with petro poroshenko, ukraine’s next president
                                                    ready; to provide the free syrian army; <with air support in fight against islamic state>
                                                    building naval bases in asia, latin america

                                                    disappointed over china’s failure; over to hand fugitive intelligence analyst edward snowden
     us        says       us          is            open; to work with new iranian president hussain rowhani
                                              not   surprised; <if north korea launches missiles>


Table 12: List of claims, or attributions by subject actors (sources) about other actors (targets). Automatically identi-
fied negative claims are emphasized.


From this set of hyperedges we extracted a directed net-                           netanyahu, uk, germany, obama} and the following ac-
work connecting actors engaging in expressions of con-                             tors remaining unassigned: {turkey, assad, the european
flict over Syria. A visualization of this network is pre-                          union}. Faction A is shown in blue in figure 6, and faction
sented in figure 6.                                                                B in red. This categorization and network visualization
     We devised a very simple algorithm to identify two fac-                       suggest that the main axis of the conflict around Syria is
tions in this conflict graph. Firstly, we attribute a score                        a Russia / U.S conflict. Factions A and B contain state ac-
s i j to every hyperedge (e i j ):                                                 tors and political leaders that are typically aligned with,
                                                                                   respectively, Russia and the U.S.
                         s i j = min(d i , d j )                                      Naturally, more sophisticated faction and alliance de-
                                                                                   tection methods can be employed. Here we are mostly
where d i is the degree (in- and out-) of node i . Then we                         interested in showing the effectiveness of our approach
iterate through hyperedges in descending order of s. This                          in summarizing complex situations from large natural
heuristic assumes that hyperedges connecting more ac-                              language corpora, and to provide some empirical valida-
tive nodes are more likely to represent the fundamental                            tion that these results are sensible. This conflict graph
dividing lines of the overall conflict. The first hyperedge                        was built from a total of 53 hyperedges, and we manually
assigns one node to faction A and another to faction B.                            verified that they all correspond to expressions of con-
From then on, a node is assigned to a faction if it does not                       flict, and that both the intervening actors and main topic
have a conflict with any current member of this faction,                           were correctly identified in all cases.
and has a conflict with a current member of the opposite
faction. In the case that the node cannot be assigned to
any faction, it remains unassigned.                                                8 Conclusions
   This resulted in faction A containing the actors: {rus-
sia, iran, moscow, putin, china, erdogan, palestinians}, fac-                      We have presented the novel SH formalism, aimed at a
tion B the actors: {us, west, israel, the united states, france,                   new approach to language understanding based on the


                                                                              26
Figure 6: Network of conflicts between actors over the topic “Syria”. Arrows point from the originator of the conflict
to its target. Size of nodes is proportional to their degree. Two factions were identified by a simple algorithm. One
faction is represented as red, the other blue. Gray nodes do not belong to any faction.


idea of translating NL into a structured and formal rep-           and finally in its ability to tackle a set of specific and re-
resentation, while allowing room for the inherent and of-          lated tasks of language understanding of particular inter-
ten irreducible ambiguities found in human communi-                est to the social sciences (actor and gender detection, co-
cation. Developing the formalism entailed an effort of             reference resolution, claim and conflict analysis), pro-
modeling of NL into a set of types and syntactic rules that        ducing results that reasonably match common-sense in-
(1) preserves the richness of NL, (2) facilitates computa-         tuitions about the ground truth, and also display good
tional language understanding tasks and (3) can act as a           precision when manually verified.
lingua franca for hybrid systems that include both hu-                The central goal has been to lay the foundations of this
man and computational intelligence of various natures              approach and demonstrate its potential. We do not claim
(e.g. symbolic, graph-based and statistical).                      to have the best performing system in any of the tasks
  Surrounding this central idea, we presented a viable             that we tackled, nor was this our aim, but we do hope
parser of NL to SH using standard contemporary ML                  to have demonstrated the versatility, completeness and
techniques as well as higher-level linguistic features pro-        potential of SH.
vided by standards NLP libraries. Furthermore, we have                Our further ambition is to apply this method to a va-
shown that inference rules and knowledge extraction                riety of language understanding tasks where the under-
patterns can be represented in SH notation itself, and we          standability of results is desirable, for example in news
have developed a procedural template for systems capa-             / social media analysis (detection of actors, topics, con-
ble of learning inference rules with the collaboration of          flicts, agreements, causality, beliefs), or to extract view-
humans and reference hypergraphs extracted from open               points from scientific articles, or to help in the study of
text.                                                              cultural objects such as literary works (known as Distant
                                                                   Reading [48]). Our hope is that we were able to con-
   We believe to have empirically validated our approach           vince the reader of the potential of SH for such tasks,
from several angles and in several ways: in terms of the           enabling large-scale text corpus analysis while preserv-
completeness of the representation by its ability to rep-          ing the rich understanding expected in social science en-
resent all grammatical constructs found in the Univer-             deavors which normally require a significant amount of
sal Dependencies; in terms of the precision of the parser          tedious manual coding [72].
across of variety of text categories; in terms of the ex-             We created the Graphbrain open-source software li-
pressive power of SH in being able to produce compet-              brary that implements all the ideas we described4 , aim-
itive results in a task for which a number of dedicated
competing systems and an external benchmark exists;                  4 https://github.com/graphbrain/graphbrain




                                                              27
ing not only to facilitate the replicability of all the ex-                   [11] Blei, D.M., 2012. Probabilistic topic models. Communica-
periments that we performed in this work, but also to                              tions of the ACM 55, 77–84.
facilitate the adoption and extension of SH and related
                                                                              [12] Blei, D.M., Ng, A.Y., Jordan, M.I., 2003. Latent dirichlet
methodology by the research community at large.
                                                                                   allocation. Journal of machine Learning research 3, 993–
                                                                                   1022.

Acknowledgements                                                              [13] Boley, H., 1977. Directed recursive labelnode hyper-
                                                                                   graphs: A new representation-language. Artificial Intel-
This research has been supported by the “Socsemics”                                ligence 9, 49–85.
Consolidator grant funded by the European Research
Council (ERC) under the European Union Horizon 2020                           [14] Bosselut, A., Rashkin, H., Sap, M., Malaviya, C., Celikyil-
research and innovation program, grant agreement No.                               maz, A., Choi, Y., 2019. Comet: Commonsense transform-
772743.                                                                            ers for automatic knowledge graph construction. arXiv
                                                                                   1906.05317.

                                                                              [15] Brown, T.B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.,
                                                                                   Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell,
References                                                                         A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan,
                                                                                   T., Child, R., Ramesh, A., Ziegler, D.M., Wu, J., Winter,
 [1] Adida, B., Birbeck, M., McCarron, S., Pemberton, S., 2008.
                                                                                   C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S.,
     RDFa in XHTML: Syntax and processing. Recommenda-
                                                                                   Chess, B., Clark, J., Berner, C., McCandlish, S., Radford,
     tion, W3C 7.
                                                                                   A., Sutskever, I., Amodei, D., 2020. Language models are
 [2] Agirre, E., Soroa, A., 2009. Personalizing pagerank for                       few-shot learners. arXiv 2005.14165.
     word sense disambiguation, in: Proceedings of the 12th
                                                                              [16] Cattuto, C., Schmitz, C., Baldassarri, A., Servedio, V.D.,
     Conference of the European Chapter of the Association
                                                                                   Loreto, V., Hotho, A., Grahl, M., Stumme, G., 2007. Net-
     for Computational Linguistics, Association for Computa-
                                                                                   work properties of folksonomies. Ai Communications 20,
     tional Linguistics. pp. 33–41.
                                                                                   245–262.
 [3] Allen, J.F., Frisch, A.M., 1982. What’s in a semantic net-
                                                                              [17] Chavalarias, D., Wallach, J.D., Li, A.H.T., Ioannidis, J.P.,
     work?, in: Proceedings of the 20th annual meeting on As-
                                                                                   2016. Evolution of reporting p values in the biomedical
     sociation for Computational Linguistics, Association for
                                                                                   literature, 1990-2015. Jama 315, 1141–1148.
     Computational Linguistics. pp. 19–27.

 [4] Angeli, G., Premkumar, M.J.J., Manning, C.D., 2015. Lever-               [18] Choi, J.D., Tetreault, J., Stent, A., 2015. It depends: Depen-
     aging linguistic structure for open domain information                        dency parser comparison using a web-based evaluation
     extraction, in: Proc. 53rd Annual Meeting of the Associa-                     tool, in: Proc. 53rd Annual Meeting of the Association for
     tion for Computational Linguistics and 7th Intl. Joint Con-                   Computational Linguistics and 7th Intl. Joint Conference
     ference on Natural Language Processing, pp. 344–354.                          on Natural Language Processing, pp. 387–396.

 [5] Angelov, D., 2020. Top2vec: Distributed representations                  [19] Collobert, R., Weston, J., 2008. A unified architecture for
     of topics. arXiv preprint arXiv:2008.09470 .                                  natural language processing: Deep neural networks with
                                                                                   multitask learning, in: Proceedings of the 25th interna-
 [6] Auer, S., Bizer, C., Kobilarov, G., Lehmann, J., Cyganiak, R.,                tional conference on Machine learning, pp. 160–167.
     Ives, Z., 2007. Dbpedia: A nucleus for a web of open data,
     in: The semantic web. Springer, pp. 722–735.                             [20] Del Corro, L., Gemulla, R., 2013. Clausie: clause-based
                                                                                   open information extraction, in: Proceedings of the 22nd
 [7] Banarescu, L., Bonial, C., Cai, S., Georgescu, M., Grif-                      international conference on World Wide Web, pp. 355–
     fitt, K., Hermjakob, U., Knight, K., Koehn, P., Palmer, M.,                   366.
     Schneider, N., 2013. Abstract meaning representation for
     sembanking, in: Proc. 7th linguistic annotation workshop                 [21] Devlin, J., Chang, M.W., Lee, K., Toutanova, K., 2019.
     and interoperability with discourse, pp. 178–186.                             BERT: Pre-training of deep bidirectional transformers for
                                                                                   language understanding, in: Proc. 2019 Conference of the
 [8] Battiston, F., Cencetti, G., Iacopini, I., Latora, V., Lucas, M.,             North American Chapter of the Association for Computa-
     Patania, A., Young, J.G., Petri, G., 2020. Networks beyond                    tional Linguistics: Human Language Technologies, ACL,
     pairwise interactions: structure and dynamics. Physics                        Minneapolis, Minnesota. pp. 4171–4186.
     Reports 874, 1–92.
                                                                              [22] Diesner, J., Carley, K., 2005. Revealing social structure
 [9] Berge, C., 1984. Hypergraphs: combinatorics of finite sets.                   from texts: Meta-matrix text analysis as a novel method
     volume 45 of North-Holland mathematical library. North-                       for network text analysis, in: Causal mapping for research
     Holland, Amsterdam.                                                           in information technology. IGI Global, pp. 81–108.

[10] Berners-Lee, T., Hendler, J., 2001. Publishing on the se-                [23] Eslahchi, C., Rahimi, A., 2007. Some properties of ordered
     mantic web. Nature 410, 1023–1024.                                            hypergraphs. Matematički vesnik 59, 9–13.


                                                                         28
[24] Etzioni, O., Banko, M., Soderland, S., Weld, D.S., 2008.               [38] Lippi, M., Torroni, P., 2016. Argumentation mining: State
     Open information extraction from the web. Communi-                          of the art and emerging trends. ACM Transactions on In-
     cations of the ACM 51, 68–74.                                               ternet Technology (TOIT) 16, 10.

[25] Fader, A., Soderland, S., Etzioni, O., 2011. Identifying re-           [39] Lowe, W., 2008. Understanding wordscores. Political Anal-
     lations for open information extraction, in: Proc. Conf. on                 ysis 16, 356–371.
     empirical methods in natural language processing, ACL.
     pp. 1535–1545.                                                         [40] Mausam, Schmitz, M., Soderland, S., Bart, R., Etzioni, O.,
                                                                                 2012. Open language learning for information extraction,
[26] Gashteovski, K., Gemulla, R., Del Corro, L., 2017. Minie:                   in: Proceedings of the 2012 joint conference on empiri-
     minimizing facts in open information extraction, in: Proc.                  cal methods in natural language processing and compu-
     of the 2017 Conf. on Empirical Methods in Natural Lan-                      tational natural language learning, pp. 523–534.
     guage Processing, Association for Computational Linguis-
                                                                            [41] Mausam, M., 2016. Open information extraction sys-
     tics. p. 2620–2630.
                                                                                 tems and downstream applications, in: Proceedings of the
[27] Gerlach, M., Peixoto, T.P., Altmann, E.G., 2018. A network                  Twenty-Fifth International Joint Conference on Artificial
     approach to topic models. Science advances 4, eaaq1360.                     Intelligence, pp. 4074–4077.

[28] Goertzel, B., 2006. Patterns, hypergraphs and embod-                   [42] McCarthy, J., 1960. Recursive functions of symbolic ex-
     ied general intelligence, in: IJCNN’06 International Joint                  pressions and their computation by machine, part i. Com-
     Conference on Neural Networks, IEEE. pp. 451–458.                           munications of the ACM 3, 184–195.

[29] Grimmer, J., Stewart, B.M., 2013. Text as data: The                    [43] Mihalcea, R., 2005. Unsupervised large-vocabulary word
     promise and pitfalls of automatic content analysis meth-                    sense disambiguation with graph-based algorithms for
     ods for political texts. Political Analysis 21, 267–297.                    sequence data labeling, in: Proceedings of Human Lan-
                                                                                 guage Technology Conference and Conference on Empiri-
[30] Hart, D., Goertzel, B., 2008. Opencog: A software frame-                    cal Methods in Natural Language Processing, pp. 411–418.
     work for integrative artificial general intelligence, in: Arti-
     ficial General Intelligence, IOS Press. pp. 468–472.                   [44] Mihalcea, R., Tarau, P., 2004. Textrank: Bringing order into
                                                                                 text, in: EMNLP’04 Proc. 2004 Conf. on Empirical Meth-
[31] Honnibal, M., Johnson, M., et al., 2015. An improved non-                   ods in Natural Language Processing, pp. 404–411.
     monotonic transition system for dependency parsing, in:
                                                                            [45] Mikolov, T., Chen, K., Corrado, G., Dean, J., 2013. Efficient
     EMNLP’15 Proc. of the 2015 Conf, on Empirical Methods
                                                                                 estimation of word representations in vector space. arXiv
     in Natural Language Processing, pp. 1373–1378.
                                                                                 preprint arXiv:1301.3781 .
[32] Iordanov, B., 2010. HyperGraphDB: A generalized graph
                                                                            [46] Miller, G.A., 1995. Wordnet: a lexical database for english.
     database, in: Shen, H.T., Pei, J., Özsu, M.T., Zou, L., Lu,
                                                                                 Communications of the ACM 38, 39–41.
     J., Ling, T.W., Yu, G., Zhuang, Y., Shao, J. (Eds.), Web-
     Age Information Management, Springer Berlin Heidel-                    [47] Monroe, B.L., Colaresi, M.P., Quinn, K.M., 2008.
     berg, Berlin, Heidelberg. pp. 25–36.                                        Fightin’words: Lexical feature selection and evalua-
                                                                                 tion for identifying the content of political conflict.
[33] Le, Q., Mikolov, T., 2014. Distributed representations of
                                                                                 Political Analysis 16, 372–403.
     sentences and documents, in: International conference
     on machine learning, PMLR. pp. 1188–1196.                              [48] Moretti, F., 2013. Distant reading. Verso Books.

[34] Léchelle, W., Gotti, F., Langlais, P., 2019. Wire57 : A fine-          [49] Murakami, K., Nichols, E., Mizuno, J., Watanabe, Y., Ma-
     grained benchmark for open information extraction, in:                      suda, S., Goto, H., Ohki, M., Sao, C., Matsuyoshi, S., Inui,
     Friedrich, A., Zeyrek, D., Hoek, J. (Eds.), Proc. of the 13th               K., et al., 2010. Statement map: reducing web information
     Linguistic Annotation Workshop, LAW at ACL 2019, Flo-                       credibility noise through opinion classification, in: Pro-
     rence, Italy, August 1, 2019, Association for Computa-                      ceedings of the fourth workshop on Analytics for noisy
     tional Linguistics. pp. 6–15.                                               unstructured text data, ACM. pp. 59–66.

[35] Léchelle, W., Gotti, F., Langlais, P., 2020. Resources for the         [50] Nadeau, D., Sekine, S., 2007. A survey of named en-
     open information extraction benchmark WiRe57, com-                          tity recognition and classification. Lingvisticae Investiga-
     panion to Léchelle et al., 2019. URL: https://github.                       tiones 30, 3–26.
     com/rali-udem/WiRe57.
                                                                            [51] Navigli, R., Lapata, M., 2007. Graph connectivity mea-
[36] Lenat, D.B., Guha, R.V., Pittman, K., Pratt, D., Shepherd,                  sures for unsupervised word sense disambiguation., in:
     M., 1990. Cyc: toward programs with common sense.                           IJCAI, pp. 1683–1688.
     Communications of the ACM 33, 30–49.
                                                                            [52] Nivre, J., De Marneffe, M.C., Ginter, F., Goldberg, Y., Ha-
[37] Leydesdorff, L., Nerghes, A., 2017. Co-word maps and                        jic, J., Manning, C.D., McDonald, R., Petrov, S., Pyysalo,
     topic modeling: A comparison using small and medium-                        S., Silveira, N., Tsarfaty, R., Zeman, D., 2016. Univer-
     sized corpora (n < 1,000). Journal of the American Society                  sal dependencies v1: A multilingual treebank collection,
     for Information Science and Technology 68, 1024–1035.                       in: Proceedings of the Tenth International Conference on


                                                                       29
     Language Resources and Evaluation (LREC’16), pp. 1659–                [67] spaCy, 2020. Models documentation. URL: https://
     1666.                                                                      spacy.io/models/en.

[53] Palmer, M., Gildea, D., Kingsbury, P., 2005. The proposi-             [68] Speer, R., Chin, J., Havasi, C., 2017. Conceptnet 5.5: An
     tion bank: An annotated corpus of semantic roles. Com-                     open multilingual graph of general knowledge, in: Pro-
     putational linguistics 31, 71–106.                                         ceedings of the AAAI Conference on Artificial Intelligence,
                                                                                pp. 4444–4451.
[54] Pang, B., Lee, L., et al., 2008. Opinion mining and sen-
                                                                           [69] Srivastava, A.N., Sahami, M., 2009. Text mining: Classifi-
     timent analysis. Foundations and Trends in Information
                                                                                cation, clustering, and applications. CRC Press.
     Retrieval 2, 1–135.
                                                                           [70] Staab, S., Studer, R., 2010. Handbook on ontologies.
[55] Peters, M.E., Neumann, M., Iyyer, M., Gardner, M., Clark,                  Springer Science & Business Media.
     C., Lee, K., Zettlemoyer, L., 2018. Deep contextualized
     word representations. arXiv 1802.05365.                               [71] Stanovsky, G., Ficler, J., Dagan, I., Goldberg, Y., 2016. Get-
                                                                                ting more out of syntax with props. arXiv 1603.01648.
[56] Reddy, S., Täckström, O., Collins, M., Kwiatkowski, T., Das,
     D., Steedman, M., Lapata, M., 2016. Transforming de-                  [72] Tilly, C., 1997. Parliamentarization of popular contention
     pendency structures to logical forms for semantic pars-                    in great britain, 1758-1834. Theory and Society 26, 245–
     ing. Transactions of the Association for Computational                     273.
     Linguistics 4, 127–140.                                               [73] Van Atteveldt, W., Kleinnijenhuis, J., Ruigrok, N., 2008.
                                                                                Parsing, semantic networks, and political authority using
[57] Ritter, A., Clark, S., Etzioni, O., et al., 2011. Named entity
                                                                                syntactic analysis to extract semantic relations from dutch
     recognition in tweets: an experimental study, in: Proceed-
                                                                                newspaper articles. Political Analysis 16, 428–446.
     ings of the Conference on Empirical Methods in Natural
     Language Processing, Association for Computational Lin-               [74] Van Atteveldt, W., Sheafer, T., Shenhav, S.R., Fogel-Dror,
     guistics. pp. 1524–1534.                                                   Y., 2017. Clause analysis: using syntactic information to
                                                                                automatically extract source, subject, and predicate from
[58] Roth, C., 2013. Socio-semantic frameworks. Advances in                     texts with an application to the 2008–2009 Gaza War. Po-
     Complex Systems 16, 1350013.                                               litical Analysis 25, 207–222.

[59] Ruiz, P., Plancq, C., Poibeau, T., 2016. More than word               [75] Vrandečić, D., 2012. Wikidata: A new platform for collab-
     cooccurrence: Exploring support and opposition in inter-                   orative data collection, in: Proceedings of the 21st inter-
     national climate negotiations with semantic parsing, in:                   national conference on World Wide Web, pp. 1063–1064.
     LREC: The 10th Language Resources and Evaluation Con-
                                                                           [76] Wang, H., Can, D., Kazemzadeh, A., Bar, F., Narayanan, S.,
     ference, pp. 1902–1907.
                                                                                2012. A system for real-time twitter sentiment analysis of
[60] Salton, G., Buckley, C., 1988. Term-weighting approaches                   2012 US presidential election cycle, in: Proceedings of the
     in automatic text retrieval. Information processing &                      ACL 2012 System Demonstrations, Association for Com-
     management 24, 513–523.                                                    putational Linguistics. pp. 115–120.

                                                                           [77] Wilkerson, J., Casas, A., 2017. Large-scale computerized
[61] Sap, M., Le Bras, R., Allaway, E., Bhagavatula, C., Lourie,
                                                                                text analysis in political science: Opportunities and chal-
     N., Rashkin, H., Roof, B., Smith, N.A., Choi, Y., 2019.
                                                                                lenges. Annual Review of Political Science 20, 529–544.
     Atomic: An atlas of machine commonsense for if-then
     reasoning, in: Proceedings of the AAAI Conference on Ar-
     tificial Intelligence, pp. 3027–3035.
                                                                           A Mapping Universal Stanford De-
[62] Shadbolt, N., Berners-Lee, T., Hall, W., 2006. The semantic
     web revisited. IEEE intelligent systems 21, 96–101.                     pendencies to hyperedges
[63] Sim, Y., Acree, B.D., Gross, J.H., Smith, N.A., 2013. Mea-            We used the Universal Stanford Dependencies [52] to
     suring ideological proportions in political speeches, in:             guide the development of the Semantic Hypergraphs rep-
     EMLP’13 Proc. 2013 Conf. on Empirical Methods in Nat-                 resentation. In table 13, we present one example of hy-
     ural Language Processing, pp. 91–101.                                 peredge for each grammatical relation in the Universal
                                                                           Dependencies. This is meant as empirical evidence for
[64] Singhal, A., 2001. Modern information retrieval: A brief
                                                                           the completeness of our model, in terms of its ability to
     overview. IEEE Data Eng. Bull. 24, 35–43.
                                                                           map to natural language constructs in most human lan-
[65] Soon, W.M., Ng, H.T., Lim, D.C.Y., 2001. A machine learn-             guages.
     ing approach to coreference resolution of noun phrases.
     Computational linguistics 27, 521–544.
                                                                           B Most common hyperedge patterns
[66] Sowa, J.F., 2014. Principles of semantic networks: Explo-
     rations in the representation of knowledge. Morgan Kauf-              Table 14 lists the 50 most commons hyperedge patterns
     mann.                                                                 in a hypergraph built from a Wikipedia sample.


                                                                      30
Grammatical Relation                 Hyperedge Example
nsubj: nominal subject               (is/P.sc (the/M dog/C) cute/C)
nsubjpass: passive nominal subject   ((was/M played/P.pa) (the/M piano/C) (by/T mary/C))
dobj: direct object (accusative)     (gave/P.sio mary/C john/C (a/M gift/C))
iobj: indirect object (dative)       (gave/P.sio mary/C john/C (a/M gift/C))
csubj: clausal subject               (makes/P.so (said/P.os what/C she/C) sense/C)
csubjpass: clausal passive subject   ((was/M suspected/P.pa) (that/T (lied/P.s she/C)) (by/T everyone/C))
ccomp: clausal complement            (says/P.so he/C (like/P.so you/C (to/M swim/C)))
xcomp: open clausal complement       (says/P.so he/C (like/P.so you/C (to/M swim/C)))
nmod: nominal modifier               ((of/B some/C (the/M toys/C))
advcl: adverbial clause modifier     (talked/P.sxx he/C (to/T him/C) (to/T (secure/P.o (the/M account/C))))
advmod: adverb modifier              ((genetically/M modified/M) food/C)
neg: negation modifier               ((not/M is/P.sc) bill/C (a/M scientist/C))
vocative: vocative                   (know/P.sv i/C john/C)
discourse: discourse                 N/A
expl: expletive                      ((there/M is/P) (a/M (in/B ghost/C (the/M room/C))))
aux: auxiliary                       ((has/M (been/M killed/P.p)) kennedy/C)
auxpass: passive auxiliary           ((has/M (been/M killed/P.p)) kennedy/C)
cop: copula                          (is/P.sc bill/C big/C)
mark: marker                         (says/P.so he/C (that/T (like/P.so you/C (to/M swim/C))))
punct: punctuation                   N/A
conj: conjunction                    (is/P.so bill/C (and/J big/C honest/C))
cc: coordination                     (is/P.so bill/C (and/J big/C honest/C))
nummod: numeric modifier             (ate/P.so sam/C (3/M sheep/C))
relcl: relative clause modifier      (saw/P.so i/C (the/M (love/P.so you/C man/C)))
det: determiner                      (is/P.sc (the/M man/C) here/C)
compound: compound                   (has/P.so john/C (the/M (+/B phone/C book/C)))
name: multi-word proper nouns        (+/B marie/C curie/C)
mwe: multi-word expression           (cried/P.sx he/C (because/T (of/M you/C)))
foreign: foreign words               misc.
goeswith: goes with                  (come/P.sox they/C here/C (without/T permission/C))
case: case marking                   (the/M (’s/B school/C grounds/C))
list: list                           (’s/B mary (:/J list/C (:/B phone/C 555-981/C) (:/B age/C 33/C)))
dislocated: dislocated elements      (is/P.scd this/C (our/M office/C) (and/J me/C sam/C))
parataxis: parataxis                 (left/P.tsi (said/P.s john/C) (the/M guy/C) (in/B early/C (the/M morning/C)))
remnant: remnant in ellipsis         (won/P.sor john/C bronze/C (:/B mary/C silver/C))
reparandum: overridden disfluency    (go/P.eo (to/T (the/M righ-/C)) (to/T (the/M left/C)))
root: sentence head                  (is/P.sc (the/M dog/C) cute/C)
dep: unspecified dependency          misc.


       Table 13: Examples of hyperedges for each grammatical relation in the Universal Dependencies.




                                                         31
                         #   Pattern                                   # Cases   OIE Pattern
                         1   (*/B.{ma} */C */C)                         32396        -
                         2   (+/B.{ma} */C */C)                         13602        2
                         3   (*/T */C)                                  12536        -
                         4   (*/B.{mm} */C */C)                          5331        -
                         5   (+/B.{mm} */C */C)                          5331        2
                         6   (*/T */R)                                   2796        -
                         7   (*/P.{so} */C */C)                          1572        1
                         8   (*/P.{sx} */C */S)                           964        3
                         9   (*/P.{sc} */C */C)                           916        1
                        10   (*/P.{sox} */C */C */S)                      861        1
                        11   (*/P.{ox} */C */S)                           631        -
                        12   (*/P.{px} */C */S)                           604        4
                        13   (*/P.{sr} */C */R)                           557        1
                        14   (*/P.{sox} */C (*/B.{ma} */C */C) */S)       337        1
                        15   (*/P.{scx} */C */C */S)                      322        1
                        16   (*/P.{so} (*/B.{ma} */C */C) */C)            305        1
                        17   (*/P.{ox} (*/B.{ma} */C */C) */S)            253        -
                        18   (*/P.{sr} */C */S)                           249        1
                        19   (*/P.{pa} */C */S)                           232        1
                        20   (*/P.{sc} (*/B.{ma} */C */C) */C)            215        1
                        21   (*/B.{aa} */C */C)                           202        -
                        22   (+/B.{aa} */C */C)                           202        -
                        23   (*/P.{sx} (*/B.{ma} */C */C) */S)            176        3
                        24   (*/P.{px} (*/B.{ma} */C */C) */S)            162        4
                        25   (*/P.{sxr} */C */S */R)                      161        3
                        26   (*/P.{sor} */C */C */R)                      158        1
                        27   (*/P.{pr} */C */R)                           150        1
                        28   (*/P.{sox} (*/B.{ma} */C */C) */C */S)       147        1
                        29   (*/P.{scx} */C (*/B.{ma} */C */C) */S)       121        5
                        30   (*/P.{ox} */C */R)                           120        -
                        31   (*/P.sr (*/B.{ma} */C */C) */R)              117        1
                        32   (*/P.{sxr} */C (*/T */C) */R)                115        3
                        33   (*/P.{scx} (*/B.{ma} */C */C) */C */S)       111        1
                        34   (*/P.{sox} */C */C */R)                      109        1
                        35   (*/P.so (+/B.{ma} */C */C) */C)               93        1
                        36   (*/P.{pax} */C */S */S)                       90        1
                        37   (*/P.{pax} */C (*/T */C) */S)                 86        1
                        38   (*/P.{sc} (*/B.{mm} */C */C) */C)             84        1
                        39   (*/P.{sc} (+/B.{mm} */C */C) */C)             84        1
                        40   (*/P.{cx} */C */S)                            81        -
                        41   (*/P.{sxr} */C */S */S)                       80        1
                        42   (*/P.{sr} (*/B.{ma} */C */C) */S)             76        1
                        43   (*/P.{sx} (+/B.{ma} */C */C) */S)             70        -
                        44   (*/P.{sxr} */C (*/T */C) */S)                 69        3
                        45   (*/P.{sc} (+/B.{ma} */C */C) */C)             69        1
                        46   (*/P.{sox} (+/B.{ma} */C */C) */C */S)        68        1
                        47   (*/P.{scx} */C */C */R)                       66        1
                        48   (*/P.{sx} */C */R)                            66        -
                        49   (*/P.{ox} (+/B.{ma} */C */C) */S)             65        -
                        50   (*/P.{or} */C */R)                            65        -



Table 14: 50 most commons hyperedge patterns found in an SH extracted from a Wikipedia sample, excluding mod-
ifiers and conjunctions and restricting expansions to depth two. Only relations with 2 or 3 arguments are accepted,
and the special builder (+/B) is explicitly considered.




                                                        32

