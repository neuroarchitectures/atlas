# Graph Matching Networks for Learning the Similarity of Graph Unknown 2019

> Source: `Graph_Matching_Networks_for_Learning_the_Similarity_of_Graph_Unknown_2019.pdf`

---

                                                                        Graph Matching Networks for
                                                             Learning the Similarity of Graph Structured Objects


                                                            Yujia Li 1 Chenjie Gu 1 Thomas Dullien 2 Oriol Vinyals 1 Pushmeet Kohli 1


                                                                 Abstract                               In the past few years graph neural networks (GNNs) have
                                                                                                        emerged as an effective class of models for learning rep-




arXiv:1904.12787v1 [cs.LG] 29 Apr 2019
                                              This paper addresses the challenging problem of           resentations of structured data and for solving various su-
                                              retrieval and matching of graph structured ob-            pervised prediction problems on graphs. Such models are
                                              jects, and makes two key contributions. First, we         invariant to permutations of graph elements by design and
                                              demonstrate how Graph Neural Networks (GNN),              compute graph node representations through a propagation
                                              which have emerged as an effective model for var-         process which iteratively aggregates local structural infor-
                                              ious supervised prediction problems defined on            mation (Scarselli et al., 2009; Li et al., 2015; Gilmer et al.,
                                              structured data, can be trained to produce embed-         2017). These node representations are then used directly for
                                              ding of graphs in vector spaces that enables effi-        node classification, or pooled into a graph vector for graph
                                              cient similarity reasoning. Second, we propose a          classification. Problems beyond supervised classification or
                                              novel Graph Matching Network model that, given            regression are relatively less well-studied for GNNs.
                                              a pair of graphs as input, computes a similarity
                                                                                                        In this paper we study the problem of similarity learning
                                              score between them by jointly reasoning on the
                                                                                                        for graph structured objects, which appears in many impor-
                                              pair through a new cross-graph attention-based
                                                                                                        tant real world applications, in particular similarity based
                                              matching mechanism. We demonstrate the ef-
                                                                                                        retrieval in graph databases. One motivating application
                                              fectiveness of our models on different domains
                                                                                                        is the computer security problem of binary function simi-
                                              including the challenging problem of control-flow-
                                                                                                        larity search, where given a binary which may or may not
                                              graph based function similarity search that plays
                                                                                                        contain code with known vulnerabilities, we wish to check
                                              an important role in the detection of vulnerabili-
                                                                                                        whether any control-flow-graph in this binary is sufficiently
                                              ties in software systems. The experimental analy-
                                                                                                        similar to a database of known-vulnerable functions. This
                                              sis demonstrates that our models are not only able
                                                                                                        helps identify vulnerable statically linked libraries in closed-
                                              to exploit structure in the context of similarity
                                                                                                        source software, a recurring problem (CVE, 2010; 2018) for
                                              learning but they can also outperform domain-
                                                                                                        which no good solutions are currently available. Figure 1
                                              specific baseline systems that have been carefully
                                                                                                        shows one example from this application, where the binary
                                              hand-engineered for these problems.
                                                                                                        functions are represented as control flow graphs annotated
                                                                                                        with assembly instructions. This similarity learning prob-
                                                                                                        lem is very challenging as subtle differences can make two
                                         1. Introduction                                                graphs be semantically very different, while graphs with
                                                                                                        different structures can still be similar. A successful model
                                         Graphs are natural representations for encoding relational
                                                                                                        for this problem should therefore (1) exploit the graph struc-
                                         structures that are encountered in many domains. Expect-
                                                                                                        tures, and (2) be able to reason about the similarity of graphs
                                         edly, computations defined over graph structured data are
                                                                                                        both from the graph structures as well as from learned se-
                                         employed in a wide variety of fields, from the analysis of
                                                                                                        mantics.
                                         molecules for computational biology and chemistry (Gilmer
                                         et al., 2017; Yan et al., 2005), to the analysis of knowl-     In order to solve the graph similarity learning problem, we
                                         edge graphs or graph structured parses for natural language    investigate the use of GNNs in this context, explore how
                                         understanding.                                                 they can be used to embed graphs into a vector space, and
                                              1
                                                                                                        learn this embedding model to make similar graphs close
                                                DeepMind 2 Google. Correspondence to: Yujia Li <yu-     in the vector space, and dissimilar graphs far apart. One
                                         jiali@google.com>.
                                                                                                        important property of this model is that, it maps each graph
                                         Proceedings of the 36 th International Conference on Machine   independently to an embedding vector, and then all the sim-
                                         Learning, Long Beach, USA, 2019. Copyright 2019 by the au-     ilarity computation happens in the vector space. Therefore,
                                         thor(s).
                                                                                                                                                                                                                                                                                              Graph Matching Networks
                                                                                                                 0xf90

                                                                                                                 push
                                                                                                                 push
                                                                                                                 mov
                                                                                                                 push
                                                                                                                 push
                                                                                                                 mov
                                                                                                                                R15 , RSP
                                                                                                                                R14 , RSP
                                                                                                                                R14D , ESI
                                                                                                                                R13 , RSP
                                                                                                                                R12 , RSP
                                                                                                                                R13D , EDI
                                                                                                                                                                                                               0x138c70

                                                                                                                                                                                                               push
                                                                                                                                                                                                               push
                                                                                                                                                                                                               push
                                                                                                                                                                                                               push
                                                                                                                                                                                                                                RBP , RSP
                                                                                                                                                                                                                                R15 , RSP
                                                                                                                                                                                                                                R14 , RSP
                                                                                                                                                                                                                                R12 , RSP
                                                                                                                                                                                                                                                                                                                                 beddings for similarity learning; (2) we propose the new
                                                                                                                                                                                                                                                                                        0x138cd3
                                                                                                                                                                                                               push             RBX , RSP
                                                                                                                 push           RBP , RSP                                                                      mov              R12 , RDX
                                                                                                                 push           RBX , RSP                                                                      mov              RBP , RDI
                                                                                                                 mov            EDI , 560                                                                      mov              RDI , [R12 + 28 ]




                                                                                                                                                                                                                                                                                                                                 Graph Matching Networks that computes similarity through
                                                                                                                                                                                                               xor              R15D , R15D
                                                                                                                 mov            EBP , EDX                                                                      test             RDI , RDI
                           0x1950                                                                                sub            RSP , 18                                                                       jz               138e4e




                                                                                        0x138d02
                                                                                                                 mov            RAX , [28]
                           push            RBP , RSP                                                             mov            [RSP + 8], RAX
                           push            R15 , RSP                                                             xor            EAX , EAX
                                                                                                                 call           f70                                                                                           0x138c8f
                           push            R14 , RSP


                                                                                                                                                                                                                                                                                        movsxd                            R14 , EAX
                                                                                                                                                                                                                              cmp            RDI , [R12 + 30 ]
                           push            R13 , RSP                                                                                                                                                                          jnbe           138e4e
                           push            R12 , RSP




                                                                                                                                                                                                                                                                                                                                 cross-graph   attention-based matching; (3) empirically we
                           push            RBX , RSP
                           push            RAX , RSP                                                             0xfc0



                                                                                                                                                                                                                                                                                        lea                               R8, RDI + R14  * 1
                                                                                                                                                                                                                              0x138c9a
                           mov             EBP , EDX


                                                                                        mov                                                                [R12 + 28 ], 0
                                                                                                                 mov            RBX , RAX
                           mov             R14D , ESI                                                            lea            EAX , R13 + f
                                                                                                                                                                                                                              mov            EBX , [R12 ]
                                                                                                                                                                                                                              mov            EDX , [RSI + 10 ]
                           mov             R15D , EDI                                                            mov            ECX , 10                                                                                      mov            R11 D, [RSI + 18 ]
                           mov             EDI , 560                                                             mov            R15D , EAX                                                                                    mov            R10 , [RSI ]
                                                                                                                                                                                                                              mov            EAX , EDX
                           call            1028                                                                  cdq            EDX , EAX



                                                                                                                                                                                                                                                                                        cmp                               R8, [R12 + 20 ]
                                                                                                                                                                                                                              shr            EAX , 3




                                                                                        jmp                                                                138e4e
                                                                                                                 and            R15D , f0                                                                                     mov            EAX , [R10 + RAX * 1 ]




                                                                                                                                                                                                                                                                                                                                 show   that the proposed graph similarity learning models
                                                                                                                 idiv           EDX , EAX , ECX                                                                               mov            ECX , EDX
                                                                                                                                                                                                                              and            CL, 7
                                                                                                                 test           RBX , RBX                                                                                     shr            EAX , CL
                                                                                                                 jz             108a                                                                                          mov            ECX , 20
                                                                                                                                                                                                                              sub            ECX , EBX




                                                                                                                                                                                                                                                                                        jbe                               138d10
                                                                                                                                                                                                                              shl/sal        EAX , CL
                           0x196d                                                                                                                                                                                             shr            EAX , CL
                                                                                                                                                                                                                              add            EDX , EBX
                                                                                                                                                                                                                              cmp            R11 D, EDX
                           mov             RBX , RAX                                                                                                                                                                          cmovbe         EDX , R11 D
                                                                                                                                       0xfdf
                           test            RBX , RBX                                                                                                                                                                          mov            [RSI + 10 ], EDX
                                                                                                                                                                                                                              test           EAX , EAX
                           jz              19ac                                                                                        lea             R12 D, RAX + 2




                                                                                                                                                                                                                                                                                                                                 achieve good performance across a range of applications,
                                                                                              0x108a                                                                                                                          jz             138d02
                                                                                                                                       lea             RAX , 2179f9
                                                                                              xor             EAX , EAX                test            BPL , 8
                                                                                                                                       mov             [RBX ], RAX
                                                                                                                                       jz              100e                                                                                           0x138cd3
                                                                                                                                                                                                       0x138d02
                                                                                                                                                                                                                                                      movsxd          R14 , EAX
                                         0x1975                                                                                                                                                         mov            [R12 + 28 ], 0                 lea             R8, RDI + R14 * 1
                                                                                                                                                                                                        jmp            138e4e                         cmp             R8, [R12 + 20 ]
                                                                                                                                                                                                                                                      jbe             138d10
                                         lea             R12 D, R15 + f




                                                                                                                                                                                                                                                                                                                                 outperforming structure agnostic models and established
                                         mov             EAX , R12 D                                                                           0xff3
   0x19ac                                sar             EAX , 1f
                                         shr             EAX , 1c                                                                              mov             EAX , EBP                                   0x138d10
    xor              EBX , EBX           lea             R13D , R15 + RAX * 1 + f             0x100e                                           and             EAX , 3                                     mov            EAX , EDX
    jmp              1a44                                                                                                                      mov             [RBX + 520 ], EAX                           shr            EAX , 3
                                         sar             R13D , 4                             mov             [RBX + 520 ], 1                  mov             EAX , EBP                                   movzx          EAX , [R10 + RAX * 1 ]                          0x138ce1
                                         lea             RAX , 21bcc9                         mov             [RBX + 524 ], 1                  sar             EAX , 4                                     xor            EBP , EBP
                                                                                                                                                                                                                                                                          lea            RDX , b18f00
                                                                                                                                                                                                           cmp            EDX , R11 D
                                         mov             [RBX ], RAX                                                                           and             EAX , 3                                     setl           BPL
                                                                                                                                                                                                                                                                          mov            ESI , 10
                                                                                                                                                                                                                                                                          xor            EAX , EAX
                                         test            BPL , 8                                                                               mov             [RBX + 524 ], EAX                           add            EBP , EDX
                                                                                                                                                                                                                                                                          mov            RDI , RBP
                                                                                                                                               jmp             1022                                        mov            [RSI + 10 ], EBP




                                                                                                                                                                                                                                                                                                                                 hand-engineered baselines.
                                         jnz             19b3                                                                                                                                              and            DL, 7
                                                                                                                                                                                                                                                                          call           67b88
                                                                                                                                                                                                           movzx          ECX , DL
                                                                                                                                                                                                           bt             EAX , ECX
                                                                                                                                                                                                           jnb            138d8a




         0x19b3
                                                        0x199b
                                                                                         0x138d10                              0x1022

                                                                                                                               bt
                                                                                                                               jnb
                                                                                                                                               EBP , 13
                                                                                                                                               1035
                                                                                                                                                                                                                             0x138d33

                                                                                                                                                                                                                             mov           EAX , EBP
         mov           EAX , EBP                                                                                                                                                                                             shr           EAX , 3
         and           EAX , 3                          mov        [RBX + 520 ], 1                                                                                                                                           mov           EDX , [R10 + RAX * 1 ]
                                                                                                                                                                                                                             mov           ECX , EBP
         mov           [RBX + 520 ], EAX                mov        EAX , 1                                                                                                                0x138d8a                           and           CL, 7                                        0x138cf7
         mov           EAX , EBP                        jmp        19c6                                                                                                                   cmp         RDI , R8
                                                                                                                                                                                                                             shr           EDX , CL
                                                                                                                                                                                                                                                                                        mov             R15D , bebbb1b7




                                                                                         mov                                                                         EAX , EDX
                                                                                                                                                                                                                             add           EBP , 4
         shr           EAX , 4                                                                         0x1035                                                                             jnb         138e4e
                                                                                                                                                                                                                             cmp           R11 D, EBP
                                                                                                                                                                                                                                                                                        jmp             138e4e

         and           EAX , 3                                                                                                                                                                                               cmovbe        EBP , R11 D
                                                                                                                                                                                                                             xor           R15D , R15D
                                                                                                       mov               EAX , EBP                      0x1028                                                               and           EDX , f
                                                                                                       shr               EAX , 1f                                                                                            mov           [RSI + 10 ], EBP
                                                                                                       bt                EBP , 1d                       call          f10                                                    jz            138e3c




                                                                                         shr                                                                         EAX , 3
                                                                                                       mov               [RBX + 514 ], EAX
                                                                                                       jnb               104d
                           0x19c6




                                                                                                                                                                                                                                                                                                                               2. Related Work
                                                                                                                                                                                                                                                                         0x138d5c




                                                                                                                                                                                                                                                                                                                             0x138ce1
                           and             R12 D, f0                                                                                                                                                                                                                     mov            EAX , EBP




                                                                                         movzx                                                                       EAX , [R10 + RAX * 1 ]
                           add             R13D , 2                                                                                                                                                                                                                      shr            EAX , 3
                                                                                                                                                                                                                                                                         movzx          EAX , [R10 + RAX * 1 ]
                           mov             [RBX + 524 ], EAX                                                                                     0x102d                                                                                                                  mov            ECX , EBP
                                                                                                0x1046
                           test            EBP , 80000                                                                                                                                                                                                                   and            CL, 7
                                                                                                                                                 mov             [RBX + 514 ], EAX                                                                                       shr            EAX , CL
                           jnz             1a1b                                                 or                [RBX + 514 ], 2                                                          0x138d93
                                                                                                                                                                                                                                  0x138e3c                               and            EAX , 1
                                                                                                                                                 jmp             1067                                                                                                    xor            ECX , ECX
                                                                                                                                                                                           lea         R9, 10e02a9




                                                                                         xor                                                                         EBP , EBP
                                                                                                                                                                                                                                  xor            EDX , EDX               cmp            EBP , R11 D
                                                                                                                                                                                           jmp         138daa
                                                                                                                                                                                                                                                                         setl           CL
                                                                                                                                                                                                                                                                         add            ECX , EBP
                                                                                                                                                                                                                                                                         mov            [RSI + 10 ], ECX




                                                                                                                                                                                                                                                                                                                             lea        RDX , b18f00
                                                                                                                                                                                     6=
                                                                                                                                                                                                                                                                         mov            ECX , EAX




                                                                                     =
                                                                                                                                                                                                                                                                         neg            ECX
                                                                                                     0x104d
                                         0x19dc                                                                                                                                                                                                                          xor            EDX , ECX




                                                                                         cmp                                                                         EDX , R11 D
                                                                                                                                                                                                                                                                         add            EDX , EAX
                                                                                                     bt                 EBP , 1e                                                                                                                                         jmp            138e3e
                                         mov             ECX , EBP                                   jnb                105a
                                         shr             ECX , 1f
                                         mov
                                         shr
                                                         EAX , EBP
                                                         EAX , 1c
                                                                                                                                                                                                 0x138daa

                                                                                                                                                                                                                                                                                                                             mov
                                                                                                                                                                                                                                                                                                                               Graph ESI    , 10
                                                                                                                                                                                                                                                                                                                                         Neural     Networks and Graph Representation
                                                                                         setl                                                                        BPL
            0x1a1b                       and             EAX , 2                                                                                                                                 movsxd          RAX , [R12 + 4]
                                                                                                                          0x1053                                                                 lea             RAX , RAX + RAX * 2
                                         lea             EDX , RAX + RCX * 1                                                                                                                     mov             RDX , [R9 + RAX * 8 + 8 ]
            call          fc0


                                                                                                                                                                                                                                                                                                                             xor        EAX , EAX
                                                                                                                                                                                                 mov             ECX , EBP
                                         test            EBP , 40000000                                                   or            [RBX + 514 ], 4                                          shr             ECX , 3
                                         lea             ECX , RAX + RCX * 1 + 4                                                                                                                 mov             EBX , [R10 + RCX * 1 ]                    0x138e3e




                                                                                         add                                                                         EBP , EDX
                                                                                                                                                                                                 mov             ECX , EBP
                                         cmovne          EDX , ECX                                                                                                                               and             CL, 7                                      movzx          ESI , DL
                                         mov             [RBX + 514 ], EDX



                                                                                                                                                                                                                                                                                                                               Learning
                                                                                                                                                                                                 shr             EBX , CL




                                                                                                                                                                                                                                                                                                                                              The history of graph neural networks (GNNs)
                                                                                                                                                                                                                                                            mov            RDX , R14
                                                                                                                                                                                                 mov             ECX , 20                                   call           67248
                                         test            EBP , 10000000


                                                                                                                                                                                                                                                                                                                             mov        RDI , RBP
                                                                                                                 0x105a                                                                          sub             ECX , [R9 + RAX * 8 ]
                                         jz              1a26                                                                                                                                    shl/sal
                                                                                                                                                                                                 shr
                                                                                                                                                                                                                 EBX , CL
                                                                                                                                                                                                                 EBX , CL
                                                                                                                  bt               EBP , 1c


                                                                                         mov                                                                         [RSI + 10 ], EBP
                                                                                                                                                                                                 movsx           EAX , [RDX + RBX * 4 + 2 ]
                                                                                                                  jnb              1067                                                          add             EAX , EBP
                                                                                                                                                                                                 cmp             R11 D, EAX
                                                                                                                                                                                                 jnbe            138de4



                                                        0x1a07
                                                                                                                                                                                                                                                                                                                             call       67b88
                                                                                         and                                                                         DL, 7
                                                                                                                          0x1060




                                                                                                                                                                                                                                                                                                                               goes back to at least the early work by Gori et al. (2005)
                                                                                                                                                                                                                 0x138de1                                   0x138e49
   0x1a20                                                test       EBP , 40000000                                        or            [RBX + 514 ], 1                                                          mov            EAX , R11 D                  add            [R12 + 28 ], R14
                                                         cmove      ECX , EAX
   mov             [RBX + 514 ], EAX                     or         ECX , 1


                                                                                         movzx                                                                       ECX , DL
                                                         mov        [RBX + 514 ], ECX
                                                                                                                                                                                                               0x138de4
                                                         jmp        1a26                                                   0x1067
                                                                                                                                                                                                               movsx            RCX , [RDX + RBX * 4 ]
                                                                                                                                                                                                               mov              [RSI + 10 ], EAX
                                                                                                                           mov            R8D, R12 D                                                           movzx            EDX , [R12 + RCX * 1 + 8 ]




                                                                                                                                                                                                                                                                                                                               and Scarselli et al. (2009), who proposed to use a propaga-
                                                                                                                           mov            ECX , R15D                                                           test             EDX , EDX




                                                                                         bt                                                                          EAX , ECX
                                                                                                                           mov            EDX , R14D                                                           jz               138e21
                                                                                                                           mov            ESI , R13D
                                  0x1a26                                                                                   mov            RDI , RBX
                                                                                                                           call           4ab0
                                  mov             RDI , RBX                                                                                                                                                                   0x138df6

                                  mov             ESI , R15D

                                                                                         jnb                                                                         138d8a
                                                                                                                                                                                                                              mov            ECX , EAX
                                                                                                                                                                                                                              shr            ECX , 3
                                  mov             EDX , R14D                                                                                                                                                                  movzx          EBP , [R10 + RCX * 1 ]
                                  mov             ECX , R12 D                                                                                                                                                                 mov            ECX , EAX




                                                                                                                                                                                                                                                                                                                               tion process to learn node representations. These models
                                                                                                                                                                                                                              and            CL, 7
                                  mov             R8D, R13D                                                         0x107b                                                                                                    shr            EBP , CL
                                  call            1a56                                                                                                                                                                        and            EBP , 1
                                                                                                                    mov              [RBX + 510 ], ffffffff                                                                   xor            ECX , ECX
                                                                                                                    mov              RAX , RBX                                                                                cmp            EAX , R11 D
                                                                                                                                                                                                                              setl           CL
                                                                                                                    jmp              108c                                                                                     add            ECX , EAX
                                                                                                                                                                                                                              mov            [RSI + 10 ], ECX
                                                                                                                                                                                                                              mov            EAX , EBP
                                                                                                                                                                                                                              neg            EAX
                           0x1a3a                                                                                                                                                                                             xor            EAX , EDX
                                                                                                                                                                                                                              add            EAX , EBP
                                                                                                           0x108c




                                                                                                                                                                                                                                                                                                                               have been further developed by incorporating modern deep
                                                                                                                                                                                                                              mov            EDX , EAX
                           mov             [RBX + 510 ], ffffffff
                                                                                                           mov             RSI , [RSP + 8]
                                                                                                           xor             RSI , [28]
                                                                                                           jz              10a1                                                                                        0x138e21

                                                                                                                                                                                                                       lea              RAX , RDI + 1
                     0x1a44                                                                                                                                                                                            mov
                                                                                                                                                                                                                       mov
                                                                                                                                                                                                                                        [R12 + 28 ], RAX
                                                                                                                                                                                                                                        [RDI ], DL
                                                                                                                                                                                                                       mov              RDI , [R12 + 28 ]
                     mov            RAX , RBX                                                                                  0x109c                                                                                  cmp              RDI , R8




                                                                                                                                                                                                                                                                                                                               learning components (Li et al., 2015; Veličković et al., 2017;
                                                                                                                                                                                                                       jb               138da0
                     add            RSP , 8
                     pop            RBX , RSP                                                                                   call            f20
                     pop            R12 , RSP
                     pop            R13 , RSP                                                                                                                                                                                                   0x138da0
                                                                                                                                                                                                              0x138e3a




                                                                                                                                                                         0x138d33
                     pop            R14 , RSP                                                                                                                                                                                                   mov                EBP , [RSI + 10 ]
                                                                                                                                                                                                              jmp              138e4e
                     pop            R15 , RSP                                                                     0x10a1
                                                                                                                                                                                                                                                mov                R11 D, [RSI + 18 ]
                                                                                                                                                                                                                                                mov                R10 , [RSI ]
                     pop            RBP , RSP
                     ret near       [RSP ]                                                                        add              RSP , 18




                                                                                                                                                                                                                                                                                                                               Bruna et al., 2013). A separate line of work focuses on gener-
                                                                                                                  pop              RBX , RSP
                                                                                                                  pop              RBP , RSP                                               0x138e4e
                                                                                                                  pop              R12 , RSP
                                                                                                                                                                                           mov         EAX , R15D
                                                                                                                  pop              R13 , RSP                                               pop         RBX , RSP
                                                                                                                  pop              R14 , RSP                                               pop         R12 , RSP
                                                                                                                  pop              R15 , RSP                                               pop         R14 , RSP




                                                                                                                                                                            mov                                  EAX , EBP
                                                                                                                                                                                           pop         R15 , RSP
                                                                                                                  ret near         [RSP ]                                                  pop         RBP , RSP
                                                                                                                                                                                           ret near    [RSP ]




                                                                                                                                                                                                                                                                                                                               alizing convolutions to graphs (Bruna et al., 2013; Bronstein
                                                                                                                                                                                                                                                                                                                               et al., 2017). Popular graph convolutional networks also
Figure 1. The binary function similarity learning problem. Check-
                                                                                                                                                                                                                                                                                                                               compute node updates by aggregating information in local
ing whether two graphs are similar requires reasoning about both
                                                                                                                                                                                                                                                                                                                               neighborhoods (Kipf & Welling, 2016), making them the
the structure as well as the semantics of the graphs. Here the left
two control flow graphs correspond to the same function compiled                                                                                                                                                                                                                                                               same family of models as GNNs. GNNs have been suc-
with different compilers (and therefore similar), while the graph                                                                                                                                                                                                                                                              cessfully used in many domains (Kipf & Welling, 2016;
on the right corresponds to a different function.                                                                                                                                                                                                                                                                              Veličković et al., 2017; Battaglia et al., 2016; 2018; Niepert
                                                                                                                                                                                                                                                                                                                               et al., 2016; Duvenaud et al., 2015; Gilmer et al., 2017; Dai
                                                                                                                                                                                                                                                                                                                               et al., 2017; Li et al., 2018; Wang et al., 2018a;b). Most of
the embeddings of graphs in a large database can be precom-                                                                                                                                                                                                                                                                    the previous work on GNNs focus on supervised prediction
puted and indexed, which enables efficient retrieval with                                                                                                                                                                                                                                                                      problems (with exceptions like (Dai et al., 2017; Li et al.,
fast nearest neighbor search data structures like k-d trees                                                                                                                                                                                                                                                                    2018; Wang et al., 2018a)). The graph similarity learning
(Bentley, 1975) or locality sensitive hashing (Gionis et al.,                                                                                                                                                                                                                                                                  problem we study in this paper and the new graph matching
1999).                                                                                                                                                                                                                                                                                                                         model can be good additions to this family of models.
We further propose an extension to GNNs which we call                                                                                                                                                                                                                                                                          Recently Xu et al. (2018) studied the discriminative power
Graph Matching Networks (GMNs) for similarity learn-                                                                                                                                                                                                                                                                           of GNNs and concluded that GNNs are as powerful as the
ing. Instead of computing graph representations indepen-                                                                                                                                                                                                                                                                       Weisfeiler-Lehman (Weisfeiler & Lehman, 1968) algorithm
dently for each graph, the GMNs compute a similarity score                                                                                                                                                                                                                                                                     in terms of distinguishing graphs (isomorphism test). In this
through a cross-graph attention mechanism to associate                                                                                                                                                                                                                                                                         paper, however we study the similarity learning problem, i.e.
nodes across graphs and identify differences. By making                                                                                                                                                                                                                                                                        how similar are two graphs, rather than whether two graphs
the graph representation computation dependent on the pair,                                                                                                                                                                                                                                                                    are identical. In this setting, learned models can adapt to the
this matching model is more powerful than the embedding                                                                                                                                                                                                                                                                        metric we have data for, while hand-coded algorithms can
model, providing a nice accuracy-computation trade-off.                                                                                                                                                                                                                                                                        not easily adapt.
We evaluate the proposed models and baselines on three                                                                                                                                                                                                                                                                         Graph Similarity Search and Graph Kernels Graph
 tasks: a synthetic graph edit-distance learning task which                                                                                                                                                                                                                                                                    similarity search has been studied extensively in database
 captures structural similarity only, and two real world tasks                                                                                                                                                                                                                                                                 and data mining communities (Yan et al., 2005; Dijkman
- binary function similarity search and mesh retrieval, which                                                                                                                                                                                                                                                                  et al., 2009). The similarity is typically defined by either ex-
 require reasoning about both the structural and semantic                                                                                                                                                                                                                                                                      act matches (full-graph or sub-graph isomorphism) (Berretti
 similarity. On all tasks, the proposed approaches outperform                                                                                                                                                                                                                                                                  et al., 2001; Shasha et al., 2002; Yan et al., 2004; Srinivasa &
 established baselines and structure agnostic models; in more                                                                                                                                                                                                                                                                  Kumar, 2003) or some measure of structural similarity, e.g.
 detailed ablation studies, we found that the Graph Matching                                                                                                                                                                                                                                                                   in terms of graph edit distances (Willett et al., 1998; Ray-
 Networks consistently outperform the graph embedding                                                                                                                                                                                                                                                                          mond et al., 2002). Most of the approaches proposed in this
 model and Siamese networks.                                                                                                                                                                                                                                                                                                   direction are not learning-based, and focus on efficiency.
To summarize, the contributions of this paper are: (1) we                                                                                                                                                                                                                                                                      Graph kernels are kernels on graphs designed to capture
demonstrate how GNNs can be used to produce graph em-
                                                   Graph Matching Networks

the graph similarity, and can be used in kernel methods for       process. Independent of our work, recently (Shyam et al.,
e.g. graph classification (Vishwanathan et al., 2010; Sher-       2017) proposed a cross-example attention model for visual
vashidze et al., 2011). Popular graph kernels include those       similarity as an alternative to Siamese networks based on
that measure the similarity between walks or paths on graphs      similar motivations and achieved good results.
(Borgwardt & Kriegel, 2005; Kashima et al., 2003; Vish-
wanathan et al., 2010), kernels based on limited-sized sub-       3. Deep Graph Similarity Learning
structures (Horváth et al., 2004; Shervashidze et al., 2009)
and kernels based on sub-tree structures (Shervashidze &          Given two graphs G1 = (V1 , E1 ) and G2 = (V2 , E2 ), we
Borgwardt, 2009; Shervashidze et al., 2011). Graph kernels        want a model that produces the similarity score s(G1 , G2 )
are usually used in models that may have learned compo-           between them. Each graph G = (V, E) is represented as
nents, but the kernels themselves are hand-designed and           sets of nodes V and edges E, optionally each node i ∈ V
motivated by graph theory. They can typically be formu-           can be associated with a feature vector xi , and each edge
lated as first computing the feature vectors for each graph       (i, j) ∈ E associated with a feature vector xij . These fea-
(the kernel embedding), and then take inner product between       tures can represent, e.g. type of a node, direction of an edge,
these vectors to compute the kernel value. One exception is       etc. If a node or an edge does not have any associated fea-
(Yanardag & Vishwanathan, 2015) where the co-occurrence           tures, we set the corresponding vector to a constant vector
of graph elements (substructures, walks, etc.) are learned,       of 1s. We propose two models for graph similarity learn-
but the basic elements are still hand-designed. Compared to       ing: a model based on standard GNNs for learning graph
these approaches, our graph neural network based similarity       embeddings, and the new and more powerful GMNs. The
learning framework learns the similarity metric end-to-end.       two models are illustrated in Figure 2.
Distance Metric Learning Learning a distance metric
between data points is the key focus of the area of metric        3.1. Graph Embedding Models
learning. Most of the early work on metric learning assumes       Graph embedding models embed each graph into a vector,
that the data already lies in a vector space, and only a linear   and then use a similarity metric in that vector space to mea-
metric matrix is learned to properly measure the distance in      sure the similarity between graphs. Our GNN embedding
this space to group similar examples together and dissimilar      model comprises 3 parts: (1) an encoder, (2) propagation
examples to be far apart (Xing et al., 2003; Weinberger &         layers, and (3) an aggregator.
Saul, 2009; Davis et al., 2007). More recently the ideas of
distance metric learning and representation learning have         Encoder The encoder maps the node and edge features to
been combined in applications like face verification, where       initial node and edge vectors through separate MLPs:
deep convolutional neural networks are learned to map sim-                   (0)
                                                                            hi = MLPnode (xi ), ∀i ∈ V
ilar images to similar representation vectors (Chopra et al.,                                                                (1)
                                                                             eij = MLPedge (xij ), ∀(i, j) ∈ E.
2005; Hu et al., 2014; Sun et al., 2014). In this paper, we
focus on representation and similarity metric learning for        Propagation Layers A propagation layer maps a set of
                                                                                           (t)
graphs, and our graph matching model goes one step beyond         node representations {hi }i∈V to new node representations
the typical representation learning methods by modeling the         (t+1)
                                                                  {hi     }i∈V , as the following:
cross-graph matchings.
                                                                                                  (t)   (t)
                                                                          mj→i = fmessage
                                                                                         (hi P, hj , eij )
Siamese Networks Siamese networks (Bromley et al.,                         (t+1)           (t)
                                                                                                                            (2)
1994; Baldi & Chauvin, 1993) are a family of neural net-                  hi     = fnode hi , j:(j,i)∈E mj→i
work models for visual similarity learning. These models
                                                                  Here fmessage is typically an MLP on the concatenated in-
typically consist of two networks with shared parameters
                                                                  puts, and fnode can be either an MLP or a recurrent neural
applied to two input images independently to compute rep-
                                                                  network core, e.g. RNN, GRU or LSTM (Li et al., 2015).
resentations, a small network is then used to fuse these
                                                                  To aggregate the messages, we use a simple sum which may
representations and compute a similarity score. They can be
                                                                  be alternatively replaced by other commutative operators
thought of as learning both the representations and the simi-
                                                                  such as mean, max or the attention-based weighted sum
larity metric. Siamese networks have achieved great success
                                                                  (Veličković et al., 2017). Through multiple layers of prop-
in many visual recognition and verification tasks (Brom-
                                                                  agation, the representation for each node will accumulate
ley et al., 1994; Baldi & Chauvin, 1993; Koch et al., 2015;
                                                                  information in its local neighborhood.
Bertinetto et al., 2016; Zagoruyko & Komodakis, 2015).
In the experiments we adapt Siamese networks to handle            Aggregator After a certain number T rounds of propaga-
graphs, but found our graph matching networks to be more          tions, an aggregator takes the set of node representations
powerful as they do cross-graph computations and therefore           (T )
                                                                  {hi } as input, and computes a graph level representa-
fuse information from both graphs early in the computation                         (T )
                                                                  tion hG = fG ({hi }). We use the following aggregation
                                                     Graph Matching Networks

                                       vector space similarity                    vector space similarity

                    graph vectors


                    propagations




                           Figure 2. Illustration of the graph embedding (left) and matching models (right).


module proposed in (Li et al., 2015),                                 graph can be matched to one or more nodes in the other:
                                                                                                 (t)       (t)
                                                                        mj→i = fmessage (hi , hj , eij ), ∀(i, j) ∈ E1 ∪ E2 (4)
                                                                                             (t)       (t)
                                                                !          µj→i = fmatch (hi , hj ),
                                      (T )               (T )
                  X
hG = MLPG               σ(MLPgate (hi ))       MLP(hi ) ,                         ∀i ∈ V1 , j ∈ V2 , or i ∈ V2 , j ∈ V1                (5)
                  i∈V                                                                                                  
                                                        (3)
                                                                         (t+1)            (t)
                                                                                              X            X
which transforms node representations and then uses a                   hi     = fnode hi ,      mj→i ,        µj 0 →i               (6)
weighted sum with gating vectors to aggregate across nodes.                                            j               j0
The weighted sum can help filtering out irrelevant informa-                               (T )
                                                                            hG1 = fG ({hi }i∈V1 )                                      (7)
tion, it is more powerful than a simple sum and also works
                                                                                        (T )
significantly better empirically.                                           hG2 = fG ({hi }i∈V2 )                                      (8)
After the graph representations hG1 and hG2 are computed                      s = fs (hG1 , hG2 ).                                     (9)
for the pair (G1 , G2 ), we compute the similarity between            Here fs is a standard vector space similarity between hG1
them using a similarity metric in the vector space, for exam-         and hG2 . fmatch is a function that communicates cross-
ple the Euclidean, cosine or Hamming similarities.                    graph information, which we propose to use an attention-
Note that without the propagation layers (or with 0 propa-            based module:
gation steps), this model becomes an instance of the Deep                                                        (t)
                                                                                                     exp(sh (hi ,hj ))
                                                                                                                            (t)

Set (Zaheer et al., 2017) or PointNet (Qi et al., 2017), which                       aj→i =        P              (t)   (t)  ,
                                                                                                     j 0 exp(sh (hi ,hj 0 ))          (10)
does computation on the individual nodes, and then pool                                                     (t)       (t)
                                                                                     µj→i =        aj→i (hi − hj )
the node representations into a representation for the whole
graph. Such a model, however, ignores the structure and               and therefore,
only treats the data as a set of independent nodes.
                                                                                            (t)  (t)    (t)         (t)
                                                                      X            X                        X
                                                                          µj→i =     aj→i (hi − hj ) = hi −   aj→i hj .
3.2. Graph Matching Networks                                           j              j                                           j
                                                                                                                                (11)
Graph matching networks take a pair of graphs as input                sh is again a vector space similarity metric, like Euclidean
and compute a similarity score between them. Compared                 or cosine similarity, aj→i are the attention weights, and
to the embedding models, these matching models compute                P                                                           (t)
                                                                         j µj→i intuitively measures the difference between hi
the similarity score jointly on the pair, rather than first inde-     and its closest neighbor in the other graph. Note that because
pendently mapping each graph to a vector. Therefore these             of the normalization in aj→i , the function fmatch implicitly
models are potentially stronger than the embedding models,                                               (t)
                                                                      depends on the whole set of {hj }, which we omitted in
at the cost of some extra computation efficiency.
                                                                      Eq. 10 for a cleaner notation. Since attention weights are
We propose the following graph matching network, which                required for every pair of nodes across two graphs, this op-
changes the node update module in each propagation layer              eration has a computation cost of O(|V1 ||V2 |), while for the
to take into account not only the aggregated messages on              GNN embedding model the cost for each round of propaga-
the edges for each graph as before, but also a cross-graph            tion is O(|V | + |E|). The extra power of the GMNs comes
matching vector which measures how well a node in one                 from utilizing the extra computation.
                                                   Graph Matching Networks

Note By construction, the attention module has a nice prop-        vectors through a tanh transformation, and optimize the
erty that, when the two graphs can be perfectly matched, and       following pair and triplet losses:
when theP  attention weights are peaked at the exact match,
we have j µj→i = 0, which means the cross-graph com-                    Lpair = E(G1 ,G2 ,t) [(t − s(G1 , G2 ))2 ]/4,   and   (14)
                                                                                                                 2
munications will be reduced to zero vectors, and the two             Ltriplet = E(G1 ,G2 ,G3 ) [(s(G1 , G2 ) − 1) +
graphs will continue to compute identical representations                                          (s(G1 , G3 ) + 1)2 ]/8,    (15)
in the next round of propagation. On the other hand, the
differences acrossPgraphs will be captured in the cross-graph                                PH
                                                                   where s(G1 , G2 ) = H1 i=1 tanh(hG1 i ) · tanh(hG2 i ) is
matching vector j µj→i , which will be amplified through           the approximate average Hamming similarity. Both losses
the propagation process, making the matching model more            are bounded in [0, 1], and they push positive pairs to have
sensitive to these differences.                                    Hamming similarity close to 1, and negative pairs to have
Compared to the graph embedding model, the matching                similarity close to -1. We found these losses to be a bit more
model has the ability to change the representation of the          stable than margin based losses for Hamming similarity.
graphs based on the other graph it is compared against.
The model will adjust graph representations to make them           4. Experiments
become more different if they do not match.
                                                                   In this section, we evaluate the graph similarity learning
3.3. Learning                                                      (GSL) framework and the graph embedding (GNNs) and
                                                                   graph matching networks (GMNs) on three tasks and com-
The proposed graph similarity learning models can be               pare these models with other competing methods. Overall
trained on a set of example pairs or triplets. Pairwise train-     the empirical results demonstrate that the GMNs excel on
ing requires us to have a dataset of pairs labeled as positive     graph similarity learning, consistently outperforming all
(similar) or negative (dissimilar), while triplet training only    other approaches.
needs relative similarity, i.e. whether G1 is closer to G2 or
G3 . We describe the losses on pairs and triplets we used be-      4.1. Learning Graph Edit Distances
low, which are then optimized with gradient descent based
algorithms.                                                        Problem Background           Graph edit distance between
                                                                   graphs G1 and G2 is defined as the minimum number of
When using Euclidean similarity, we use the following              edit operations needed to transform G1 to G2 . Typically
margin-based pairwise loss:                                        the edit operations include add/remove/substitute nodes and
                                                                   edges. Graph edit distance is naturally a measure of simi-
 Lpair = E(G1 ,G2 ,t) [max{0, γ − t(1 − d(G1 , G2 ))}], (12)
                                                                   larity between graphs and has many applications in graph
where t ∈ {−1, 1} is the label for this pair, γ > 0 is a           similarity search (Dijkman et al., 2009; Zeng et al., 2009;
margin parameter, and d(G1 , G2 ) = khG1 − hG2 k2 is the           Gao et al., 2010). However computing the graph edit dis-
Euclidean distance. This loss encourages d(G1 , G2 ) <             tance is NP-hard in general (Zeng et al., 2009), therefore
1−γ when the pair is similar (t = 1), and d(G1 , G2 ) > 1+γ        approximations have to be used. Through this experiment
when t = −1. Given triplets where G1 and G2 are closer             we show that the GSL models can learn structural similarity
than G1 and G3 , we optimize the following margin-based            between graphs on very challenging problems.
triplet loss:                                                  Training Setup We generated training data by sampling
                                                               random binomial graphs G1 with n nodes and edge prob-
Ltriplet = E(G1 ,G2 ,G3 ) [max{0, d(G1 , G2 )−d(G1 , G3 )+γ}].
                                                               ability p (Erdös & Rényi, 1959), and then create positive
                                                          (13)
                                                               example G2 by randomly substituting kp edges from G1
This loss encourages d(G1 , G2 ) to be smaller than
                                                               with new edges, and negative example G3 by substituting
d(G1 , G3 ) by at least a margin γ.
                                                               kn edges from G1 , where kp < kn 1 . A model needs to
For applications where it is necessary to search through       predict a higher similarity score for positive pair (G1 , G2 )
a large database of graphs with low latency, it is benefi-     than negative pair (G1 , G3 ). Throughout the experiments
cial to have the graph representation vectors be binary, i.e.  we fixed the dimensionality of node vectors to 32, and the
hG ∈ {−1, 1}H , so that efficient nearest neighbor search      dimensionality of graph vectors to 128 without further tun-
algorithms (Gionis et al., 1999) may be applied. In such       ing. We also tried different number of propagation steps T
cases, we can minimize the Hamming distance of positive           1
pairs and maximize it for negative pairs. With this restric-        Note that even though G2 is created with kp edge substitutions
                                                               from G1 , the actual edit-distance between G1 and G2 can be
tion the graph vectors can no longer freely occupy the whole   smaller than kp due to symmetry and isomorphism, same for G3
Euclidean space, but we gain the efficiency for fast retrieval and kn . However the probability of such cases is typically low and
and indexing. To achieve this we propose to pass the hG        decreases rapidly with increasing graph sizes.
                                                    Graph Matching Networks

  Graph Distribution   WL kernel       GNN           GMN           4.2. Control Flow Graph based Binary Function
   n = 20, p = 0.2     80.8 / 83.2   88.8 / 94.0   95.0 / 95.6          Similarity Search
   n = 20, p = 0.5     74.5 / 78.0   92.1 / 93.4   96.6 / 98.0
   n = 50, p = 0.2     93.9 / 97.8   95.9 / 97.2   97.4 / 97.6     Problem Background Binary function similarity search
   n = 50, p = 0.5     82.3 / 89.0   88.5 / 91.0   93.8 / 92.6
                                                                   is an important problem in computer security. The need
                                                                   to analyze and search through binaries emerges when we
Table 1. Comparing the graph embedding (GNN) and matching          do not have access to the source code, for example when
(GMN) models trained on graphs from different distributions with
                                                                   dealing with commercial or embedded software or suspi-
the baseline, measuring pair AUC / triplet accuracy (×100).
                                                                   cious executables. Combining a disassembler and a code
from 1 to 5, and observed consistently better performance          analyzer, we can extract a control-flow graph (CFG) which
with increasing T . The results reported in this section are       contains all the information in a binary function in a struc-
all with T = 5 unless stated otherwise. More details are           tured format. See Figure 1 and Appendix B.2 for a few
included in Appendix B.1.                                          example CFGs. In a CFG, each node is a basic block of as-
                                                                   sembly instructions, and the edges between nodes represent
Baseline We compare our models with the popular We-                the control flow, indicated by for example a jump or a return
isfeiler Lehman (WL) kernel (Shervashidze et al., 2011),           instruction used in branching, loops or function calls. In this
which has been shown to be very competitive on graph clas-         section, we target the vulnerability search problem, where a
sification tasks and the Weisfeiler Lehman algorithm behind        piece of binary known to have some vulnerabilities is used
this kernel is a strong method for checking graph isomor-          as the query, and we search through a library to find similar
phism (edit distance of 0), a closely related task (Weisfeiler     binaries that may have the same vulnerabilities.2 Accurate
& Lehman, 1968; Shervashidze et al., 2011).                        identification of similar vulnerabilities enables security en-
Evaluation       The performance of different models are           gineers to quickly narrow down the search space and apply
evaluated using two metrics: (1) pair AUC - the area under         patches.
the ROC curve for classifying pairs of graphs as similar or        In the past the binary function similarity search problem
not on a fixed set of 1000 pairs and (2) triplet accuracy -        has been tackled with classical graph theoretical matching
the accuracy of correctly assigning higher similarity to the       algorithms (Eschweiler et al., 2016; Pewny et al., 2015),
positive pair in a triplet than the negative pair on a fixed set   and Xu et al. (2017) and Feng et al. (2016) proposed to
of 1000 triplets.                                                  learn embeddings of CFGs and do similarity search in the
Results We trained and evaluated the GSL models on                 embedding space. Xu et al. (2017) in particular proposed an
graphs of a few specific distributions with different n, p,        embedding method based on graph neural networks, starting
with kp = 1 and kn = 2 fixed. The evaluation results are           from some hand selected feature vectors for each node. Here
shown in Table 1. We can see that by learning on graphs            we study further the performance of graph embedding and
of specific distributions, the GSL models are able to do           matching models, with pair and triplet training, different
better than generic baselines, and the GMNs consistently           number of propagation steps, and learning node features
outperform the embedding model (GNNs). Note that this              from the assembly instructions.
result does not contradict with the conclusion by Xu et al.        Training Setup and Baseline We train and evaluate our
(2018), as we are learning a similarity metric, rather than        model on data generated by compiling the popular open
doing an isomorphism test, and our model can do better than        source video processing software ffmpeg using different
the WL-kernel with learning.                                       compilers gcc and clang, and different compiler opti-
For the GMNs, we can visualize the cross-graph attention to        mization levels, which results in 7940 functions and roughly
gain further insight into how it is working. Figure 3 shows        8 CFGs per function. The average size of the CFGs is
two examples of this for a matching model trained with n           around 55 nodes per graph, with some larger graphs having
sampled from [20, 50], tested on graphs of 10 nodes. The           up to a few thousand nodes (see Appendix B.2 for more
cross-graph attention weights are shown in green, with the         detailed statistics). Different compiler optimization levels
scale of the weights shown as the transparency of the green        result in CFGs of very different sizes for the same function.
edges. We can see that the attention weights can align nodes       We split the data and used 80% functions and the associated
well when the two graphs match, and tend to focus on nodes         CFGs for training, 10% for validation and 10% for testing.
with higher degrees when they don’t. However the pattern           The models were trained to learn a similarity metric on
is not as interpretable as in standard attention models.           CFGs such that the CFGs for the same function have high
                                                                   similarity, and low similarity otherwise. Once trained, this
More experiments on generalization capabilities of these           similarity metric can be used to search through library of
models (train on small graphs, test on larger graphs, train
                                                                       2
on graphs with some kp , kn combinations, test on others)                Note that our formulation is general and can also be applied
                                                                   to source code directly if they are available.
and visualizations are included in Appendix B.1 and C.
                                                     Graph Matching Networks




                    graph edit distance = 1                                             graph edit distance = 2
Figure 3. Visualization of cross-graph attention for GMNs after 5 propagation layers. In each pair of graphs the left figure shows the
attention from left graph to the right, the right figure shows the opposite.

binaries and be invariant to compiler type and optimization          the graph structure, and it achieved 0.619 AUC and 24.5%
levels.                                                              triplet accuracy. This is not surprising as the WL kernel is
                                                                     not designed for solving this task, while our models learn
We compare our graph embedding and matching models
                                                                     the features useful for the task of interest, and can achieve
with Google’s open source function similarity search tool
                                                                     better performance than generic similarity metrics.
(Dullien, 2018), which has been used to successfully find
vulnerabilities in binaries in the past. This tool computes
representations of CFGs through a hand-engineered graph              4.3. More Baselines and Ablation Studies
hashing process which encodes the neighborhood structure             In this section, we carefully examine the effects of the de-
of each node by hashing the degree sequence from a traver-           sign decisions we made in the GMN model and compare it
sal of a 3-hop neighborhood, and also encodes the assembly           against a few more alternatives. In particular, we evaluate
instructions for each basic block by hashing the trigrams of         the popular Graph Convolutional Network (GCN) model
assembly instruction types. These features are then com-             by Kipf & Welling (2016) as an alternative to our GNN
bined by using a SimHash-style (Charikar, 2002) algorithm            model, and Siamese versions of the GNN/GCN embedding
with learned weights to form a 128-dimensional binary code.          models. The GCN model replaces the message passing in
An LSH-based search index is then used to perform approx-            Eq. 2 with graph convolutions, and the Siamese model pre-
imate nearest neighbor search using hamming distance.                dicts a distance value by concatenating two graph vectors
Following (Dullien, 2018), we also map the CFGs to 128-              and then pass through a 2 layer MLP. The comparison with
dimensional binary vectors, and use the Hamming similarity           Siamese networks can in particular show the importance of
formulation described in Section 3 for training. We further          the cross-graph attention early on in the similarity computa-
studied two variants of the data, one that only uses the graph       tion process, as Siamese networks fuse the representations
structure, and one that uses both the graph structure and            for 2 graphs only at the very end.
the assembly instructions with learned node features. When           We focus on the function similarity search task, and also
assembly instructions are available, we embed each instruc-          conduct experiments on an extra COIL-DEL mesh graph
tion type into a vector, and then sum up all the embedding           dataset (Riesen & Bunke, 2008), which contains 100 classes
vectors for instructions in a basic block as the initial repre-      of mesh graphs corresponding to 100 types of objects. We
sentation vector (the xi ’s) for each node, these embeddings         treat graphs in the same class as similar, and used identical
are learned jointly with the rest of the model.                      setup as the function similarity search task for training and
Results       Figure 4 shows the performance of different            evaluation.
models with different number of propagation steps and in             Table 2 summarizes the experiment results, which clearly
different data settings. We again evaluate the performance           show that: (1) the GNN embedding model is a competi-
of these models on pair AUC and triplet accuracy on fixed            tive model (more powerful than the GCN model); (2) using
sets of pairs and triplets from the test set. It is clear from re-   Siamese network architecture to learn similarity on top of
sults that: (1) the performance of both the graph embedding          graph representations is better than using a prespecified sim-
and matching models consistently go up with more propaga-            ilarity metric (Euclidean, Hamming etc.); (3) the GMNs
tion steps, and in particular significantly outperforming the        outperform the Siamese models showing the importance of
structure agnostic model special case which uses 0 propaga-          cross-graph information communication early in the compu-
tion steps; (2) the graph embedding model is consistently            tation process.
better than the baselines with enough propagation steps;
and (3) graph matching models outperforms the embedding
models across all settings and propagation steps. Addition-
ally, we have tried the WL kernel on this task using only
                                                          Graph Matching Networks
                            100                                                           100

                            95                                                            95



                                                                       Triplet Accuracy
                 Pair AUC
                                                                                          90
                            90                                                                              baseline: struct only
                                                                                                            baseline: struct + node features
                                                                                                            embedding: struct only
                                                                                          85                embedding: struct + node features
                            85                                                                              matching: struct only
                                                                                                            matching: struct + node features
                                                                                          80
                                  0   1   2    3     4     5   6   7                            0   1   2       3       4      5      6         7
                                          #propagation steps                                            #propagation steps
                    Figure 4. Performance (×100) of different models on the binary function similarity search task.

                      Model Pair AUC Triplet Acc
                                                                                                 Model           Pair AUC          Triplet Acc
                    Baseline    96.09         96.35
                                                                                                  GCN              94.80              94.95
                       GCN      96.67         96.57
                                                                                          Siamese-GCN              95.90              96.10
               Siamese-GCN      97.54         97.51
                                                                                                  GNN              98.58              98.70
                       GNN      97.71         97.83
                                                                                          Siamese-GNN              98.76              98.55
               Siamese-GNN      97.76         97.58
                                                                                                 GMN               98.97              98.80
                      GMN       99.28         99.18
                     Function Similarity Search                                                              COIL-DEL

                        Table 2. More results on the function similarity search task and the extra COIL-DEL dataset.

5. Conclusions and Discussion                                                      setting together with a faster filtering model like the graph
                                                                                   embedding model or standard graph similarity search meth-
In this paper we studied the problem of graph similarity                           ods, to narrow down the search to a smaller candidate set,
learning using graph neural networks. Compared to standard                         and then use the more expensive matching model to rerank
prediction problems for graphs, similarity learning poses a                        the candidates to improve precision.
unique set of challenges and potential benefits. For exam-
ple, the graph embedding models can be learned through a                           Developing neural models for graph similarity learning is an
classification setting when we do have a set of classes in the                     important research direction with many applications. There
dataset, but formulating it as a similarity learning problem                       are still many interesting challenges to resolve, for example
can handle cases where we have a very large number of                              to improve the efficiency of the matching models, study
classes and only very few examples for each class. The                             different matching architectures, adapt the GNN capacity to
representations learned from the similarity learning setting                       graphs of different sizes, and applying these models to new
can also easily generalize to data from classes unseen during                      application domains. We hope our work can spur further
training (zero-shot generalization).                                               research in this direction.
We proposed the new graph matching networks as a stronger
alternative to the graph embedding models. The added                               References
power for the graph matching models comes from the fact                            CVE-2010-0188.  Available from MITRE, CVE-
that they are not independently mapping each graph to an                            ID CVE-2010-0188., 2010.      URL https:
embedding, but rather doing comparisons at all levels across                        //cve.mitre.org/cgi-bin/cvename.cgi?
the pair of graphs, in addition to the embedding computa-                           name=cve-2010-0188.
tion. The model can then learn to properly allocate capacity
toward the embedding part or the matching part. The price                          CVE-2018-0986.  Available from MITRE, CVE-
to pay for this expressivity is the added computation cost                          ID CVE-2018-0986., 2018.      URL https:
in two aspects: (1) since each cross-graph matching step re-                        //cve.mitre.org/cgi-bin/cvename.cgi?
quires the computation of the full attention matrices, which                        name=CVE-2018-0986.
requires at least O(|V1 ||V2 |) time, this may be expensive for
large graphs; (2) the matching models operate on pairs, and                        Baldi, P. and Chauvin, Y. Neural networks for fingerprint
cannot directly be used for indexing and searching through                           recognition. Neural Computation, 5(3):402–418, 1993.
large graph databases. Therefore it is best to use the graph
matching networks when we (1) only care about the similar-                         Battaglia, P., Pascanu, R., Lai, M., Rezende, D. J., and
ity between individual pairs, or (2) use them in a retrieval                         Kavukcuoglu, K. Interaction networks for learning about
                                                 Graph Matching Networks

  objects, relations and physics. In Advances in neural         Dai, H., Khalil, E., Zhang, Y., Dilkina, B., and Song,
  information processing systems, pp. 4502–4510, 2016.            L. Learning combinatorial optimization algorithms over
                                                                  graphs. In Advances in Neural Information Processing
Battaglia, P. W., Hamrick, J. B., Bapst, V., Sanchez-             Systems, pp. 6351–6361, 2017.
  Gonzalez, A., Zambaldi, V., Malinowski, M., Tacchetti,
  A., Raposo, D., Santoro, A., Faulkner, R., Gulcehre, C.,      Davis, J. V., Kulis, B., Jain, P., Sra, S., and Dhillon, I. S.
  Song, F., Ballard, A., Gilmer, J., Dahl, G., Vaswani,           Information-theoretic metric learning. In Proceedings of
  A., Allen, K., Nash, C., Langston, V., Dyer, C., Heess,         the 24th international conference on Machine learning
  N., Wierstra, D., Kohli, P., Botvinick, M., Vinyals, O.,       (ICML), pp. 209–216, 2007.
  Li, Y., and Pascanu, R. Relational inductive biases,          Dijkman, R., Dumas, M., and García-Bañuelos, L. Graph
  deep learning, and graph networks. arXiv preprint               matching algorithms for business process model simi-
  arXiv:1806.01261, 2018.                                         larity search. In International Conference on Business
Bentley, J. L. Multidimensional binary search trees used for      Process Management, pp. 48–63. Springer, 2009.
  associative searching. Communications of the ACM, 18          Dullien, T. functionsimsearch. https://github.com/
  (9):509–517, 1975.                                              google/functionsimsearch, 2018. Accessed:
                                                                  2018-05-14.
Berretti, S., Del Bimbo, A., and Vicario, E. Efficient match-
  ing and indexing of graph models in content-based re-         Duvenaud, D. K., Maclaurin, D., Iparraguirre, J., Bombarell,
  trieval. IEEE Transactions on Pattern Analysis and Ma-          R., Hirzel, T., Aspuru-Guzik, A., and Adams, R. P. Con-
  chine Intelligence, 23(10):1089–1105, 2001.                     volutional networks on graphs for learning molecular fin-
                                                                  gerprints. In Advances in neural information processing
Bertinetto, L., Valmadre, J., Henriques, J. F., Vedaldi, A.,      systems, pp. 2224–2232, 2015.
  and Torr, P. H. Fully-convolutional siamese networks for
  object tracking. In European conference on computer           Erdös, P. and Rényi, A. On random graphs, i. Publicationes
  vision, pp. 850–865. Springer, 2016.                            Mathematicae (Debrecen), 6:290–297, 1959.
                                                                Eschweiler, S., Yakdan, K., and Gerhards-Padilla, E. dis-
Borgwardt, K. M. and Kriegel, H.-P. Shortest-path kernels
                                                                  covre: Efficient cross-architecture identification of bugs
  on graphs. In Data Mining, Fifth IEEE International
                                                                  in binary code. In NDSS, 2016.
  Conference on, pp. 8–pp. IEEE, 2005.
                                                                Feng, Q., Zhou, R., Xu, C., Cheng, Y., Testa, B., and Yin,
Bromley, J., Guyon, I., LeCun, Y., Säckinger, E., and Shah,       H. Scalable graph-based bug search for firmware images.
  R. Signature verification using a" siamese" time delay          In Proceedings of the 2016 ACM SIGSAC Conference on
  neural network. In Advances in Neural Information Pro-          Computer and Communications Security, pp. 480–491.
  cessing Systems, pp. 737–744, 1994.                             ACM, 2016.
Bronstein, M. M., Bruna, J., LeCun, Y., Szlam, A., and Van-     Gao, X., Xiao, B., Tao, D., and Li, X. A survey of graph
  dergheynst, P. Geometric deep learning: going beyond            edit distance. Pattern Analysis and applications, 13(1):
  euclidean data. IEEE Signal Processing Magazine, 34(4):        113–129, 2010.
  18–42, 2017.
                                                                Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and
Bruna, J., Zaremba, W., Szlam, A., and LeCun, Y. Spec-            Dahl, G. E. Neural message passing for quantum chem-
  tral networks and locally connected networks on graphs.         istry. arXiv preprint arXiv:1704.01212, 2017.
  arXiv preprint arXiv:1312.6203, 2013.                         Gionis, A., Indyk, P., Motwani, R., et al. Similarity search
Charikar, M. S. Similarity estimation techniques from round-      in high dimensions via hashing. In Vldb, pp. 518–529,
  ing algorithms. In Proceedings of the Thiry-fourth Annual       1999.
  ACM Symposium on Theory of Computing, STOC ’02,               Gori, M., Monfardini, G., and Scarselli, F. A new model for
  pp. 380–388, New York, NY, USA, 2002. ACM. ISBN 1-              learning in graph domains. In IEEE International Joint
  58113-495-9. doi: 10.1145/509907.509965. URL http:              Conference on Neural Networks (IJCNN), volume 2, pp.
  //doi.acm.org/10.1145/509907.509965.                            729–734. IEEE, 2005.
Chopra, S., Hadsell, R., and LeCun, Y. Learning a sim-          Horváth, T., Gärtner, T., and Wrobel, S. Cyclic pattern ker-
  ilarity metric discriminatively, with application to face       nels for predictive graph mining. In Proceedings of the
  verification. In Computer Vision and Pattern Recognition,       tenth ACM SIGKDD international conference on Knowl-
  2005. CVPR 2005. IEEE Computer Society Conference               edge discovery and data mining, pp. 158–167. ACM,
  on, volume 1, pp. 539–546. IEEE, 2005.                          2004.
                                                 Graph Matching Networks

Hu, J., Lu, J., and Tan, Y.-P. Discriminative deep metric      Shasha, D., Wang, J. T., and Giugno, R. Algorithmics and
  learning for face verification in the wild. In Proceedings     applications of tree and graph searching. In Proceedings
  of the IEEE Conference on Computer Vision and Pattern          of the twenty-first ACM SIGMOD-SIGACT symposium on
 Recognition, pp. 1875–1882, 2014.                               Principles of database systems, pp. 39–52. ACM, 2002.

Kashima, H., Tsuda, K., and Inokuchi, A. Marginalized ker-     Shervashidze, N. and Borgwardt, K. M. Fast subtree kernels
  nels between labeled graphs. In Proceedings of the 20th        on graphs. In Advances in neural information processing
  international conference on machine learning (ICML-03),        systems, pp. 1660–1668, 2009.
  pp. 321–328, 2003.
                                                               Shervashidze, N., Vishwanathan, S., Petri, T., Mehlhorn, K.,
Kingma, D. P. and Ba, J. Adam: A method for stochastic           and Borgwardt, K. Efficient graphlet kernels for large
  optimization. arXiv preprint arXiv:1412.6980, 2014.            graph comparison. In Artificial Intelligence and Statistics,
Kipf, T. N. and Welling, M. Semi-supervised classifica-          pp. 488–495, 2009.
  tion with graph convolutional networks. arXiv preprint
                                                               Shervashidze, N., Schweitzer, P., Leeuwen, E. J. v.,
  arXiv:1609.02907, 2016.
                                                                 Mehlhorn, K., and Borgwardt, K. M. Weisfeiler-lehman
Koch, G., Zemel, R., and Salakhutdinov, R. Siamese neural        graph kernels. Journal of Machine Learning Research,
  networks for one-shot image recognition. In ICML Deep          12(Sep):2539–2561, 2011.
  Learning Workshop, volume 2, 2015.
                                                               Shyam, P., Gupta, S., and Dukkipati, A. Attentive recurrent
Li, Y., Tarlow, D., Brockschmidt, M., and Zemel, R.              comparators. arXiv preprint arXiv:1703.00767, 2017.
  Gated graph sequence neural networks. arXiv preprint
  arXiv:1511.05493, 2015.                                      Srinivasa, S. and Kumar, S. A platform based on the multi-
                                                                 dimensional data model for analysis of bio-molecular
Li, Y., Vinyals, O., Dyer, C., Pascanu, R., and Battaglia,       structures. In Proceedings of the VLDB Conference, pp.
  P. Learning deep generative models of graphs. arXiv            975–986. Elsevier, 2003.
  preprint arXiv:1803.03324, 2018.
                                                               Sun, Y., Chen, Y., Wang, X., and Tang, X. Deep learning
Niepert, M., Ahmed, M., and Kutzkov, K. Learning con-            face representation by joint identification-verification. In
  volutional neural networks for graphs. In International        Advances in neural information processing systems, pp.
  conference on machine learning, pp. 2014–2023, 2016.           1988–1996, 2014.
Pewny, J., Garmany, B., Gawlik, R., Rossow, C., and Holz,
                                                               Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,
  T. Cross-architecture bug search in binary executables.
                                                                 L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Atten-
  In IEEE Symposium on Security and Privacy (SP), pp.
                                                                 tion is all you need. In Advances in Neural Information
  709–724. IEEE, 2015.
                                                                 Processing Systems, pp. 6000–6010, 2017.
Qi, C. R., Su, H., Mo, K., and Guibas, L. J. Pointnet: Deep
  learning on point sets for 3d classification and segmen-     Veličković, P., Cucurull, G., Casanova, A., Romero, A.,
  tation. In IEEE Conference on Computer Vision and              Liò, P., and Bengio, Y. Graph attention networks. arXiv
  Pattern Recognition (CVPR), 2017.                              preprint arXiv:1710.10903, 2017.

Raymond, J. W., Gardiner, E. J., and Willett, P. Rascal:       Vishwanathan, S. V. N., Schraudolph, N. N., Kondor, R.,
  Calculation of graph similarity using maximum common           and Borgwardt, K. M. Graph kernels. Journal of Machine
  edge subgraphs. The Computer Journal, 45(6):631–644,           Learning Research, 11(Apr):1201–1242, 2010.
  2002.
                                                               Wang, T., Liao, R., Ba, J., and Fidler, S. Nervenet: Learning
Riesen, K. and Bunke, H. Iam graph database repository          structured policy with graph neural networks. In ICLR,
  for graph based pattern recognition and machine learn-        2018a.
  ing. In Joint IAPR International Workshops on Statistical
  Techniques in Pattern Recognition (SPR) and Structural       Wang, Y., Sun, Y., Liu, Z., Sarma, S. E., Bronstein, M. M.,
  and Syntactic Pattern Recognition (SSPR), pp. 287–297.        and Solomon, J. M. Dynamic graph cnn for learning on
  Springer, 2008.                                               point clouds. arXiv preprint arXiv:1801.07829, 2018b.

Scarselli, F., Gori, M., Tsoi, A. C., Hagenbuchner, M., and    Weinberger, K. Q. and Saul, L. K. Distance metric learning
  Monfardini, G. The graph neural network model. IEEE           for large margin nearest neighbor classification. Journal
  Transactions on Neural Networks, 20(1):61–80, 2009.           of Machine Learning Research, 10(Feb):207–244, 2009.
                                               Graph Matching Networks

Weisfeiler, B. and Lehman, A. A reduction of a graph
 to a canonical form and an algebra arising during this
 reduction. Nauchno-Technicheskaya Informatsia, 2(9):
 12–16, 1968.
Willett, P., Barnard, J. M., and Downs, G. M. Chemical
 similarity searching. Journal of chemical information
 and computer sciences, 38(6):983–996, 1998.

Xing, E. P., Jordan, M. I., Russell, S. J., and Ng, A. Y.
  Distance metric learning with application to clustering
  with side-information. In Advances in neural information
  processing systems, pp. 521–528, 2003.

Xu, K., Hu, W., Leskovec, J., and Jegelka, S. How
  powerful are graph neural networks? arXiv preprint
  arXiv:1810.00826, 2018.
Xu, X., Liu, C., Feng, Q., Yin, H., Song, L., and Song,
  D. Neural network-based graph embedding for cross-
  platform binary code similarity detection. In Proceedings
  of the 2017 ACM SIGSAC Conference on Computer and
 Communications Security, pp. 363–376. ACM, 2017.
Yan, X., Yu, P. S., and Han, J. Graph indexing: a frequent
  structure-based approach. In Proceedings of the ACM
  SIGMOD international conference on Management of
  data, pp. 335–346. ACM, 2004.
Yan, X., Yu, P. S., and Han, J. Substructure similarity
  search in graph databases. In Proceedings of the ACM
  SIGMOD international conference on Management of
  data, pp. 766–777, 2005.

Yanardag, P. and Vishwanathan, S. Deep graph kernels.
  In Proceedings of the 21th ACM SIGKDD International
  Conference on Knowledge Discovery and Data Mining,
  pp. 1365–1374, 2015.
Zagoruyko, S. and Komodakis, N. Learning to compare
  image patches via convolutional neural networks. In IEEE
  Conference on Computer Vision and Pattern Recognition
  (CVPR), pp. 4353–4361. IEEE, 2015.
Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B.,
  Salakhutdinov, R. R., and Smola, A. J. Deep sets. In
  Advances in Neural Information Processing Systems, pp.
  3394–3404, 2017.
Zeng, Z., Tung, A. K., Wang, J., Feng, J., and Zhou, L.
  Comparing stars: On approximating graph edit distance.
  Proceedings of the VLDB, 2(1):25–36, 2009.
                                                   Graph Matching Networks

A. Extra Details on Model Architectures                           for the edit distance learning task without further tuning.
                                                                  Using larger models there should further improve model
In the propagation layers of the graph embedding and match-       performance.
ing models, we used an MLP with one hidden layer as the
fmessage module, with a ReLU nonlinearity on the hidden
                                     (t)                          B.1. Learning Graph Edit Distances
layer. For node state vectors (the hi vectors) of dimension
D, the size of the hidden layer and the output is set to 2D.      In this task the nodes and edges have no extra features
We found it to be beneficial to initialize the weights of this    associated with them, we therefore initialized the xi and xij
fmessage module to be small, which helps stablizing train-        vectors as vectors of 1s, and the encoder MLP in Eq.1 is
ing. We used the standard Glorot initialization with an extra     simply a linear layer for the nodes and an identity mapping
scaling factor of 0.1. When not using this small scaling          for the edges.
factor, at the begining of training the message vectors when      We searched through the following hyperparameters: (1)
summed up can have huge scales, which is bad for learning.        triplet vs pair training; (2) number of propagation layers;
One extra thing to note about the propagation layers is that      (3) share parameters on different propagation layers or not.
we can make all the propagation layers share the same set of      Learning rate is fixed at 0.001 for all runs and we used the
parameters, which can be useful if this is a suitable inductive   Adam optimizer (Kingma & Ba, 2014). Overall we found:
bias to have.                                                     (1) triplet and pair training performs similarly, with pair
                                                                  training slightly better, (2) using more propagation layers
We tried different fnode modules in both experiments, and         consistently helps, and increasing the number of propaga-
found GRUs to generally work better than one-hidden layer         tion layers T beyond 5 may help even more, (3) sharing
                              P uses GRUs as fnode , with
MLPs, and all the results reported                                parameters is useful for performance more often than not.
the sum over edge messages j mj→i treated as the input
to the                                                            Intuitively, the baseline WL kernel starts by labeling each
    P GRU for the Pembedding model, and the concatenation
of j mj→i and j 0 µj 0 →i as the input to the GRU for             node by its degree, and then iteratively updates a node’s
the matching model.                                               representation as the histogram of neighbor node patterns,
                                                                  which is effectively also a graph propagation process. The
In the aggregator module, we used a single linear layer for       kernel value is then computed as a dot product of graph rep-
the node transformation MLP and the gating MLPgate in             resentation vectors, which is the histogram of different node
Eq.3. The output of this linear layer has a dimensionality the    representations. When using the kernel with T iterations
same as the required dimensionality for the graph vectors.        of computation, a pair of graphs of size |V | can have as
σ(x) = 1+e1−x is the logistic sigmoid function, and is            large as a 2|V |T dimensional representation vector for each
the element-wise product. After the weighted sum, another         graph, and these sets of effective ‘feature’ types are differ-
MLP with one hidden layers is used to further transform           ent for different pairs of graphs as the node patterns can be
the graph vector. The hidden layer has the same size as the       very different. This is an advantage for WL kernel over our
output, with a ReLU nonlinearity.                                 models as we used a fixed sized graph vector regardless of
For the matching model, the attention weights are computed        the graph size. We evaluate WL kernel for T up to 5 and
as                                                                report results for the best T on the evaluation set.
                                  (t)     (t)
                      exp(sh (hi , hj ))                          In addition to the experiments presented in the main paper,
            aj→i = P                              .   (16)
                                      (t)     (t)                 we have also tested the generalization capabilities of the
                       j 0 exp(sh (hi , hj 0 ))
                                                                  proposed models, and we present the extra results in the
We have tried the Euclidean similarity sh (hi , hj ) =            following.
−khi − hj k2 for sh , as well as the dot-product similar-         Train on small graphs, generalize to large graphs. In
ity sh (hi , hj ) = h>
                     i hj , and they perform similarly without    this experiment, we trained the GSL models on graphs with
significant difference.                                           n sampled uniformly from 20 to 50, and p sampled from
                                                                  range [0.2, 0.5] to cover more variability in graph sizes and
B. Extra Experiment Details                                       edge density for better generalization, and we again fix kp =
                                                                  1, kn = 2. For evaluation, we tested the best embedding
We fixed the node state vector dimensionality to 32, and          models and matching models on graphs with n = 100, 200
graph vector dimensionality to 128 throughout both the            and p = 0.2, 0.5, with results shown in Table 3. We can see
graph edit distance learning and binary function similar-         that for this task the GSL models trained on small graphs
ity search tasks. We tuned this initially on the function         can generalize to larger graphs than they are trained on.
similarity search task, which clearly performs better than        The performance falls off a bit on much larger graphs with
smaller models. Increasing the model size however leads to        much more nodes and edges. This is also partially caused
overfitting for that task. We directly used the same setting
                                                     Graph Matching Networks

    Eval Graphs        WL kernel         GNN           GMN           Overall we found that (1) triplet training performs slightly
  n = 100, p = 0.2     98.5 / 99.4     96.6 / 96.8   96.8 / 97.7     better than pair training in this case; (2) both learning rates
  n = 100, p = 0.5     86.7 / 97.0     79.8 / 81.4   83.1 / 83.6
  n = 200, p = 0.2     99.9 / 100.0    88.7 / 88.5   89.4 / 90.0     can work but the smaller learning rate is more stable; (3)
  n = 200, p = 0.5     93.5 / 99.2     72.0 / 72.3   68.3 / 70.1     increasing number of propagation layers generally helps; (4)
                                                                     using different propagation layer parameters perform better
Table 3. Generalization performance on large graphs for the GSL      than using shared parameters; (5) GRUs are more stable
models trained on small graphs with 20 ≤ n ≤ 50 and 0.2 ≤ p ≤        than MLPs and performs overall better.
0.5.
                                                                     In addition to the results reported in the main paper, we have
                                                                     also tried the same models on another dataset obtained by
                                                                     compiling the compression software unrar with different
by the fact that we are using a fixed sized graph vector             compilers and optimization levels. Our graph similarity
throughout the experiments , but the WL kernel on the                learning methods also perform very well on the unrar data,
other hand has much more effective ‘features’ to use for             but this dataset is a lot smaller, with around 400 functions
computing similarity. On the other hand, as shown before,            only, and overfitting is therefore a big problem for any learn-
when trained on graphs from distributions we care about,             ing based model, so the results on this dataset are not very
the GSL models can adapt and perform much better.                    reliable to draw any conclusions.
Train on some kp , kn combinations, test on other combi-             A few more control-flow graph examples are shown in Fig-
nations. We have also tested the model trained on graphs             ure 5. The distribution of graph sizes in the training set is
with n ∈ [20, 50], p ∈ [0.2, 0.5], kp = 1, kn = 2, on graphs         shown in Figure 6.
with different kp and kn combinations. In particular, when
evaluated on kp = 1, kn = 4, the models perform much
better than on kp = 1, kn = 2, reaching 1.0 AUC and                  C. Extra Attention Visualizations
100% triplet accuracy easily, as this is considerably sim-           A few more attention visualizations are included in Figure 7,
pler than the kp = 1, kn = 2 setting. When evaluated on              Figure 8 and Figure 9. Here the graph matching model
graphs with kp = 2, kn = 3, the performance is workse                we used has shared parameters for all the propagation and
than kp = 1, kn = 2 as this is a harder setting.                     matching layers and was trained with 5 propagation layers.
In addition, we have also tried training on the more diffi-          Therefore we can use a number T different from the num-
cult setting kp = 2, kn = 3, and evaluate the models on              ber of propagation layers the model is being trained on to
graphs with kp = 1, kn = 2 and n ∈ [20, 50], p ∈ [0.2, 0.5].         test the model’s performance. In both visualizations, we
The performance of the models on these graphs are ac-                unrolled the propagation for up to 9 steps and the model still
tually be better than the models trained on this setting             computes sensible attention maps even with T > 5.
of kp = 1, kn = 2, which is surprising and clearly                   Note that the attention maps do not converge to very peaked
demonstrates the value of good training data. However,               distributions. This is partially due to the fact that we used
in terms of generalizing to larger graphs models trained on          the node state vectors both to carry information through the
kp = 2, kn = 3 does not have any significant advantages.             propagation process, as well as in the attention mechanism
                                                                     as is. This makes it hard for the model to have very peaked
B.2. Binary Function Similarity Search                               attention as the scale of these node state vectors won’t be
In this task the edges have no extra features so we initialize       very big. A better solution is to compute separate key, query
them to constant vectors of 1s, and the encoder MLP for              and value vectors for each node as done in the tensor2tensor
the edges is again just an identity mapping. When using the          self-attention formulation (Vaswani et al., 2017), which may
CFG graph structure only, the nodes are also initialized to          further improve the performance of the matching model.
constant vectors of 1s, and the encoder MLP is a linear layer.       Figure 7 shows another possibility where the attention maps
In the case when using assembly instructions, we have a list         do not converge to very peaked distributions because of
of assembly code associated with each node. We extracted             in-graph symmetries. Such symmetries are very typical
the operator type (e.g. add, mov, etc.) from each instruc-           in graphs. In this case even though the attention maps
tion, and then embeds each operator into a vector, the initial       are not peaked, the cross graph communication vectors µ
node representation is a sum of all operator embeddings.             are still zero, and the two graphs will still have identical
We searched through the following hyperparameters: (1)               representation vectors.
triplet or pair training, (2) learning rate in {10−3 , 10−4 }, (3)
number of propagation layers; (4) share propagation layer
parameters or not; (5) GRU vs one-layer MLP for the fnode
module.
                                                                                                                                                     Graph Matching Networks




                                                                                                                             0x1ceba

                                                                                                                             sub           RSP , 38
                                                                                                                             mov           [RSP + c], EDI
                                                                                                                             mov           [RSP + 8], ESI
                                                                                                                             mov           [RSP + 4], EDX
                                                                                                                             mov           RAX , [28]
                                                                                                                             mov           [RSP + 28 ], RAX
                                                                                                                             xor           EAX , EAX
                                                                                                                             mov           EDI , 560
                                                                                                                             call          fb8




                                                                                                                             0x1cee4

                                                                                                                             mov           [RSP + 20 ], RAX
                                                                                                                             mov           EAX , [RSP + c]
                                                                                                                             add           EAX , f
                                                                                                                             and           EAX , f0
                                                                                                                             mov           [RSP + 18 ], EAX
                                                                                                                             mov           EAX , [RSP + c]
                                                                                                                             add           EAX , f
                                                                                                                             lea           EDX , RAX + f
                                                                                                                             test          EAX , EAX
                                                                                                                             cmovs         EAX , EDX
                                                                                                                             sar           EAX , 4
                                                                                                                             add           EAX , 2
                                                                                                                             mov           [RSP + 1c ], EAX
                                                                                                                             cmp           [RSP + 20 ], 0
                                                                                                                             jnz           1cf22




                                                                                                        0x1cf22
                                                                                                                                                                                                                                                                                                                       0xf90
                                                                                                         mov             RAX , [RSP + 20 ]               0x1cf18
                                                                                                         lea             RDX , 220cd9                                                                                                                                                                                  push           R15 , RSP
                                                                                                         mov             [RAX ], RDX                     mov           EAX , 0                                                                                                                                         push           R14 , RSP
                                                                                                         mov             EAX , [RSP + 4]                 jmp           1d088                                                                                                                                           mov            R14D , ESI
                                                                                                         and             EAX , 8                                                                                                                                                                                       push           R13 , RSP
                                                                                                         test            EAX , EAX                                                                                                                                                                                     push           R12 , RSP
                                                                                                         jz              1cf69                                                                                                                                                                                         mov            R13D , EDI
                                                                                                                                                                                                                                                                                                                       push           RBP , RSP
                                                                                                                                                                                                                                                                                                                       push           RBX , RSP
                                                                                                                                                                                                                                                                                                                       mov            EDI , 560
                                                                                                                                                                                                                                                                                                                       mov            EBP , EDX
                                                                                                                                         0x1cf3c                                                                                                                                                                       sub            RSP , 18
                                                                                                                                         mov              EAX , [RSP + 4]
                                                                                                                                                                                                                                                                                                                       mov            RAX , [28]
                                                                                                                                         and              EAX , 3                                                                                                                                                      mov            [RSP + 8], RAX
                                                                                                                                         mov              EDX , EAX                                                                                                                                                    xor            EAX , EAX
                                                                                             0x1cf69                                                                                                                                                                                                                   call           f70
                                                                                                                                         mov              RAX , [RSP + 20 ]
                                                                                             mov          RAX , [RSP + 20 ]              mov              [RAX + 520 ], EDX
                                                                                             mov          [RAX + 520 ], 1                mov              EAX , [RSP + 4]
                                                                                             mov          RAX , [RSP + 20 ]              sar              EAX , 4
                                                                                             mov          [RAX + 524 ], 1                and              EAX , 3
                                                                                                                                         mov              EDX , EAX                                                                                                                                                    0xfc0
                                                                                                                                         mov              RAX , [RSP + 20 ]
                                                                                                                                         mov              [RAX + 524 ], EDX                                                                                                                                            mov            RBX , RAX
                                                                                                                                         jmp              1cf87                                                                                                                                                        lea            EAX , R13 + f
                                                                                                                                                                                                                                                                                                                       mov            ECX , 10
                                                                                                                                                                                                                                                                                                                       mov            R15D , EAX
                                                                                                                                                                                                                                                                                                                       cdq            EDX , EAX
                                                                                                                                                                                                                                                                                                                       and            R15D , f0
                                                                                                                                                                                                                                                                                                                       idiv           EDX , EAX , ECX
                                                                                                                  0x1cf87
                                                                                                                                                                                                                                                                                                                       test           RBX , RBX
                                                                                                                  mov            EAX , [RSP + 4]                                                                                                                                                                       jz             108a
                                                                                                                  and            EAX , 80000
                                                                                                                  test           EAX , EAX
                                                                                                                  jz             1cfab

                                                                                                                                                                                                                                                                                                                                             0xfdf

                                                                                                                                                                                                                                                                                                    0x108a                                   lea             R12 D, RAX + 2
                                                                                             0x1cfab                                                                                                                                                                                                                                         lea             RAX , 2179f9
                                                                                                                                         0x1cf94                                                                                                                                                    xor             EAX , EAX                test            BPL , 8
                                                                                             mov          RAX , [RSP + 20 ]                                                                                                                                                                                                                  mov             [RBX ], RAX
                                0x1950                                                       mov          [RAX + 514 ], 0                call             f50                                                                                                                                                                                jz              100e
                                                                                             cmp          [RSP + 4], 0
                                push            RBP , RSP                                    jns          1cfdc                                                                                                             0x1752b
                                push            R15 , RSP                                                                                                                                                                   push             R15 , RSP
                                push            R14 , RSP                                                                                                                                                                   push             R14 , RSP
                                push            R13 , RSP                                                                                                                                                                   push             R13 , RSP
                                push            R12 , RSP                                                                                                                                                                   push             R12 , RSP                                                                                               0xff3
                                push            RBX , RSP                                     0x1cfc1                                                                                                                       push             RBP , RSP
                                push            RAX , RSP                                                                                      0x1cf99                                                                      push             RBX , RSP                                                                                               mov             EAX , EBP
                                                                                              mov              RAX , [RSP + 20 ]                                                                                            sub              RSP , 18                                                                                                and             EAX , 3
                                mov             EBP , EDX                                                                                                                                                                   mov              R12 D, EDI                                             0x100e
                                mov             R14D , ESI                                    mov              EAX , [RAX + 514 ]              mov              EDX , EAX                                                                                                                                                                            mov             [RBX + 520 ], EAX
                                                                                                                                                                                                                            mov              R13D , ESI
                                mov             R15D , EDI                                    or               EAX , 1                         mov              RAX , [RSP + 20 ]                                           mov              EBP , EDX                                              mov             [RBX + 520 ], 1                  mov             EAX , EBP
                                mov             EDI , 560                                     mov              EDX , EAX                       mov              [RAX + 514 ], EDX                                           mov              RAX , [28]                                             mov             [RBX + 524 ], 1                  sar             EAX , 4
                                call            1028                                          mov              RAX , [RSP + 20 ]               jmp              1d054                                                       mov              [RSP + 8], RAX                                                                                          and             EAX , 3
                                                                                              mov              [RAX + 514 ], EDX                                                                                            xor              EAX , EAX                                                                                               mov             [RBX + 524 ], EAX
                                                                                                                                                                                                                            mov              EDI , 560                                                                                               jmp             1022
                                                                                                                                                                                                                            call             f38


                                0x196d                                                       0x1cfdc
                                mov             RBX , RAX                                                                                                                                                                    0x1755b                                                                                                 0x1022
                                                                                              mov          EAX , [RSP + 4]
                                test            RBX , RBX                                     and          EAX , 20000000                                                                                                    mov              RBX , RAX
                                jz              19ac                                                                                                                                                                         lea              EAX , R12 + f                                                                          bt              EBP , 13
                                                                                              test         EAX , EAX
                                                                                                                                                                                                                             mov              R14D , EAX                                                                             jnb             1035
                                                                                              jz           1d004
                                                                                                                                                                                                                             and              R14D , f0
                                                                                                                                                                                                                             mov              ECX , 10
                                                                                                                                                                                                                             cdq              EDX , EAX
                                                                                                                                                                                                                             idiv             EDX , EAX , ECX
                                              0x1975                                                                                                                                                                         lea              R15D , RAX + 2
                                                                                                                   0x1cfe9                                                                                                   mov              EAX , 0                                                        0x1035
                                              lea             R12 D, R15 + f                                                                                                                                                 test             RBX , RBX
                                              mov             EAX , R12 D                                          mov              RAX , [RSP + 20 ]                                                                        jz               175db                                                          mov               EAX , EBP
        0x19ac                                                                                                     mov              EAX , [RAX + 514 ]                                                                                                                                                                                                        0x1028
                                              sar             EAX , 1f                                                                                                                                                                                                                                       shr               EAX , 1f
                                              shr             EAX , 1c                                             or               EAX , 2                                                                                                                                                                  bt                EBP , 1d
         xor              EBX , EBX                                                                                mov              EDX , EAX                                                                                                                                                                                                                 call          f10
                                              lea             R13D , R15 + RAX * 1 + f                                                                                                                                                                                                                       mov               [RBX + 514 ], EAX
         jmp              1a44                sar             R13D , 4                                             mov              RAX , [RSP + 20 ]                                                                                                0x17580
                                                                                                                                                                                                     0x175db                                                                                                 jnb               104d
                                              lea             RAX , 21bcc9                                         mov              [RAX + 514 ], EDX
                                                                                                                                                                                                                                                     lea              RAX , 21aa19
                                              mov             [RBX ], RAX                                                                                                                             mov             RSI , [RSP + 8]
                                                                                                                                                                                                                                                     mov              [RBX ], RAX
                                              test            BPL , 8                                                                                                                                 xor             RSI , [28]
                                                                                                                                                                                                                                                     test             BPL , 8
                                                                                                                                                                                                      jz              1763a
                                              jnz             19b3                                                                                                                                                                                   jz               175f0

                                                                                                        0x1d004                                                                                                                                                                                                                                        0x102d
                                                                                                                                                                                                                                                                                                      0x1046
                                                                                                        mov            EAX , [RSP + 4]                                                                                                                                                                                                                 mov             [RBX + 514 ], EAX
                                                                                                        and            EAX , 40000000                                                                                                                                                                 or                [RBX + 514 ], 2
                                                                                                                                                                                    0x1763a
                                                                                                                                                                                                                                                                 0x17590                                                                               jmp             1067
              0x19b3                                                                                    test           EAX , EAX
                                                             0x199b                                     jz             1d02c                                                        add         RSP , 18
                                                                                                                                                                                                                                                                 mov            EAX , EBP
              mov           EAX , EBP                                                                                                                                               pop         RBX , RSP
                                                                                                                                                                                                                      0x175eb                                    and            EAX , 3
              and           EAX , 3                          mov        [RBX + 520 ], 1                                                                                             pop         RBP , RSP
                                                                                                                                                                                                                                                                 mov            [RBX + 520 ], EAX
              mov           [RBX + 520 ], EAX                                                                                                                                       pop         R12 , RSP
                                                             mov        EAX , 1                                                                                                                                       call             ee8                       mov            EAX , EBP
              mov           EAX , EBP                                                                                                                                               pop         R13 , RSP                                                                                                  0x104d
                                                             jmp        19c6                                                                                                                                                                                     sar            EAX , 4
                                                                                                                                                                                    pop         R14 , RSP
              shr           EAX , 4                                                                                                                                                                                                                              and            EAX , 3
                                                                                                                   0x1d011                                                          pop         R15 , RSP
              and           EAX , 3                                                                                                                                                 ret near    [RSP ]
                                                                                                                                                                                                                                                                 mov            [RBX + 524 ], EAX          bt                 EBP , 1e
                                                                                                                   mov              RAX , [RSP + 20 ]
                                                                                                                                                                                                                                                                                                           jnb                105a
                                                                                                                   mov              EAX , [RAX + 514 ]
                                                                                                                   or               EAX , 4
                                                                                                                   mov              EDX , EAX
                                0x19c6                                                                                                                                                                               0x175f0
                                                                                                                   mov              RAX , [RSP + 20 ]
                                                                                                                                                                                                                                                                                                                                0x1053
                                                                                                                   mov              [RAX + 514 ], EDX                                                                mov             [RBX + 520 ], 1
                                and             R12 D, f0
                                                                                                                                                                                                                     mov             [RBX + 524 ], 1
                                add             R13D , 2                                                                                                                                                                                                                                                                        or            [RBX + 514 ], 4
                                                                                                                                                                                                                     jmp             175a9
                                mov             [RBX + 524 ], EAX
                                test            EBP , 80000
                                jnz             1a1b                                                    0x1d02c

                                                                                                        mov            EAX , [RSP + 4]                                                                                       0x175a9                                                                                   0x105a
                                                                                                        and            EAX , 10000000                                                                                        bt               EBP , 13
                                                                                                        test           EAX , EAX                                                                                                                                                                                        bt               EBP , 1c
                                                                                                                                                                                                                             jnb              17606
                                                                                                        jz             1d054                                                                                                                                                                                            jnb              1067
                                              0x19dc

                                              mov             ECX , EBP
                                              shr             ECX , 1f
                                              mov             EAX , EBP                                                                                                                        0x17606
                                                                                                                                                                                                                                                                                                                                0x1060
                                              shr             EAX , 1c                                             0x1d039                                                                     mov             EAX , EBP                        0x175af
                 0x1a1b                       and             EAX , 2                                                                                                                          shr             EAX , 1f                                                                                                         or            [RBX + 514 ], 1
                                              lea             EDX , RAX + RCX * 1                                  mov              RAX , [RSP + 20 ]                                          mov             [RBX + 514 ], EAX                 call           ee0
                 call          fc0                                                                                 mov              EAX , [RAX + 514 ]
                                              test            EBP , 40000000                                                                                                                   bt              EBP , 1d
                                              lea             ECX , RAX + RCX * 1 + 4                              or               EAX , 1                                                    jnb             1761e
                                              cmovne          EDX , ECX                                            mov              EDX , EAX
                                              mov             [RBX + 514 ], EDX                                    mov              RAX , [RSP + 20 ]
                                                                                                                   mov              [RAX + 514 ], EDX                                                                                                                                                                            0x1067
                                              test            EBP , 10000000
                                              jz              1a26                                                                                                                              0x17617                                          0x175b4                                                                         mov            R8D, R12 D
                                                                                                                                                                                                                                                                                                                                 mov            ECX , R15D
                                                                                                                                                                                                or              [RBX + 514 ], 2                   mov           [RBX + 514 ], EAX
                                                                                                                                                                                                                                                                                                                                 mov            EDX , R14D
                                                                                                                   0x1d054                                                                                                                                                                                                       mov            ESI , R13D
                                                                                                                                                                                                                                                                                                                                 mov            RDI , RBX
                                                             0x1a07                                                mov              EDI , [RSP + 1c ]                                                                                                                                                                            call           4ab0
                                                                                                                                                                                                     0x1761e
                                                                                                                   mov              ECX , [RSP + 18 ]
        0x1a20                                                test       EBP , 40000000                            mov              EDX , [RSP + 8]                                                  bt              EBP , 1e
                                                              cmove      ECX , EAX                                 mov              ESI , [RSP + c]                                                  jnb             1762b
        mov             [RBX + 514 ], EAX                     or         ECX , 1                                   mov              RAX , [RSP + 20 ]
                                                              mov        [RBX + 514 ], ECX                         mov              R8D, EDI                                                                                                                                                                              0x107b
                                                              jmp        1a26                                      mov              RDI , RAX
                                                                                                                   call             1cc23                                                                              0x17624                                                                                            mov              [RBX + 510 ], ffffffff
                                                                                                                                                                                                                                                                                                                          mov              RAX , RBX
                                                                                                                                                                                                                       or                [RBX + 514 ], 4                                                                  jmp              108c
                                       0x1a26
                                                                                                                0x1d074
                                       mov             RDI , RBX
                                                                                                                                                                                                               0x1762b
                                       mov             ESI , R15D                                               mov             RAX , [RSP + 20 ]                                                                                                                                                                0x108c
                                       mov             EDX , R14D                                               mov             [RAX + 510 ], ffffffff                                                         bt               EBP , 1c
                                       mov             ECX , R12 D                                              mov             RAX , [RSP + 20 ]                                                              jnb              175ba                                                                            mov             RSI , [RSP + 8]
                                       mov             R8D, R13D                                                                                                                                                                                                                                                 xor             RSI , [28]
                                       call            1a56                                                                                                                                                                                                                                                      jz              10a1

                                                                                                                                                                                                                       0x17631
                                                                                                                          0x1d088
                                                                                                                                                                                                                       or                [RBX + 514 ], 1
                                0x1a3a                                                                                    mov            RCX , [RSP + 28 ]                                                             jmp               175ba                                                                                       0x109c
                                                                                                                          xor            RCX , [28]
                                mov             [RBX + 510 ], ffffffff                                                                                                                                                                                                                                                                call            f20
                                                                                                                          jz             1d09d


                                                                                                                                                                                                                            0x175ba

                          0x1a44                                                                                                                                                                                            mov              R8D, R15D
                                                                                                                                 0x1d098                                                                                    mov              ECX , R14D                                                                 0x10a1
                                                                                                                                                                                                                            mov              EDX , R13D
                          mov            RAX , RBX                                                                               call           f60                                                                         mov              ESI , R12 D                                                                add              RSP , 18
                          add            RSP , 8                                                                                                                                                                            mov              RDI , RBX
                          pop            RBX , RSP                                                                                                                                                                          call             61e4
                                                                                                                                                                                                                                                                                                                        pop              RBX , RSP
                          pop            R12 , RSP                                                                                                                                                                                                                                                                      pop              RBP , RSP
                          pop            R13 , RSP                                                                                                                                                                                                                                                                      pop              R12 , RSP
                          pop            R14 , RSP                                                                    0x1d09d                                                                                                                                                                                           pop              R13 , RSP
                          pop            R15 , RSP                                                                                                                                                                                                                                                                      pop              R14 , RSP
                                                                                                                                                                                                                               0x175ce                                                                                  pop              R15 , RSP
                          pop            RBP , RSP                                                                    add            RSP , 38
                          ret near       [RSP ]                                                                       ret near       [RSP ]                                                                                    mov             [RBX + 510 ], ffffffff                                                   ret near         [RSP ]
                                                                                                                                                                                                                               mov             RAX , RBX




Figure 5. Example control flow graphs for the same binary function, compiled with different compilers (clang for the leftmost one, gcc
for the others) and optimization levels. Note that each node in the graphs also contains a set of assembly instructions which we also take
into account when computing similarity using learned features.
                                                                                           Graph Matching Networks




                          10 4



                          10 3




             Graph size
                          10 2



                          10 1



                          10 0 0   10000   20000      30000        40000   50000   60000
                                               Graphs sorted by size



Figure 6. Control flow graph size distribution in the training set. In
this plot the graphs are sorted by size on the x axis, each point in
the figure corresponds to the size of one graph.
                                                     Graph Matching Networks




                        0 propagation steps                                              1 propagation step




                        2 propagation steps                                              3 propagation step




                        4 propagation steps                                              5 propagation step




                        6 propagation steps                                              7 propagation step




                        8 propagation steps                                             9 propagation steps

Figure 7. The change of cross-graph attention over propagation layers. Here the two graphs are two isomorphic chains and there are some
in-graph symmetries. Note that in the end the nodes are matched to two corresponding nodes with equal weight, except the one at the
center of the chain which can only match to a single other node.
                                                    Graph Matching Networks




                        0 propagation steps                                             1 propagation step




                        2 propagation steps                                             3 propagation step




                        4 propagation steps                                             5 propagation step




                        6 propagation steps                                             7 propagation step




                        8 propagation steps                                            9 propagation steps

Figure 8. The change of cross-graph attention over propagation layers. Here the two graphs are isomorphic, with graph edit distance 0.
Note that in the end a lot of the matchings concentrated on the correct match.
                                             Graph Matching Networks




                0 propagation steps                                             1 propagation step




                2 propagation steps                                             3 propagation step




                4 propagation steps                                             5 propagation step




                6 propagation steps                                             7 propagation step




                8 propagation steps                                             9 propagation steps

Figure 9. The change of cross-graph attention over propagation layers. The edit distance between these two graphs is 1.

