# Hyperbolic graph embeddings Graph embedding setup Hamilton Hamilton 2019

> Source: `Hyperbolic_graph_embeddings_Graph_embedding_setup_Hamilton_Hamilton_2019.pdf`

---

Hyperbolic graph embeddings
William L. Hamilton
Visiting Researcher, FAIR Montreal; Assistant Professor (2019), McGill
Based on: Nickel and Kiela. Poincaré embeddings for learning
hierarchical representations. NIPS, 2017.


                        William L. Hamilton, McGill University, MILA, FAIR   1
Graph embedding setup
• Encode nodes so that distance in the embedding
  space approximates similarity in the original
  graph/network.




                     William L. Hamilton, McGill University, MILA, FAIR   2
 Focus on simple encoders (no GCNs)
§ Encoder is just an “embedding-lookup”
§ I.e., one embedding per node.
                        embedding vector for a
      embedding            specific node
        matrix

                                                                       Dimension/size
           Z=                                                          of embeddings



                  one column per node
                  William L. Hamilton, McGill University, MILA, FAIR                    3
Simple example: edge likelihood from distance
• Encode nodes so that distance in the embedding
  space approximates similarity in the original
  graph/network.                edge probability

                                                                   1
            P ((u, v) = 1) =
                                                    ed(u,v) + 1
                              Distance in the
                             embedding space


                    William L. Hamilton, McGill University, MILA, FAIR   4
What kind of embedding space?

                                                                      ?
                                                                  ?
                                                                              ? ?
                                                        ?             ?           ?
                                                             ?
                                                                                      ?
                                                                              ?
                                                                          ?       ?



             William L. Hamilton, McGill University, MILA, FAIR                           5
What kind of embedding space?
• Standard practice is to use Euclidean space and
  distance function, e.g.,


• But what about non-Euclidean spaces, such as
  hyperbolic geometries?
• Might there be a benefit to embedding in those spaces?

                   William L. Hamilton, McGill University, MILA, FAIR   6
       Euclidean vs. hyperbolic geometry

Euclidean                                        Hyperbolic
§   No intrinsic curvature                       §       Constant negative
§   Parallel lines always same                           curvature.
    distance.                                    §       Parallel lines diverge.
§   Pythagorean theorem                          §       No Pythagorean theorem.




                     William L. Hamilton, McGill University, MILA, FAIR        7
Euclidean vs. hyperbolic geometry
 Grids are naturally embedding in                          Trees are naturally embedded in
         Euclidean space.                                         hyperbolic space.




                          William L. Hamilton, McGill University, MILA, FAIR                 8
Models of hyperbolic space
§ There are many different ways to represent
  hyperbolic space.
§ Two of the most popular are:
   § Poincare disk/ball
       § Points are embedded within closed unit-sphere.
       § Distances grow towards the boundary and become infinite.
       § (Conformal with Euclidean space)
   § Lorenz model
       § Points are embedded in surface of hyperboloid.
                      William L. Hamilton, McGill University, MILA, FAIR   9
Models of hyperbolic space
      Poincare disk/ball                                                Lorenz model




                   William L. Hamilton, McGill University, MILA, FAIR                  10
Distance in hyperbolic space
• Distances grow exponentially as we increase vector
  norms (i.e., asmove to the outside of the disk).
• More “precision” as we move towards the boundary.




                     William L. Hamilton, McGill University, MILA, FAIR   11
Distance in hyperbolic space
• Hyperbolic distance naturally captures graph
  distance for trees/hierarchies.
  “distance ratio”
     d(x, y)         1. Consider two children of a node in a tree.
d(x, O) + d(y, O)    2. Graph “distance ratio” (left) is always 1.
        O
                     3. What happens when we embed them?


   x          y

                       William L. Hamilton, McGill University, MILA, FAIR   12
Distance in hyperbolic space
• Hyperbolic distance naturally captures graph
  distance for trees/hierarchies.
  “distance ratio”
     d(x, y)
d(x, O) + d(y, O)
        O


   x          y

                      William L. Hamilton, McGill University, MILA, FAIR   13
Distance in hyperbolic space
As we move towards edge of the disk, hyperbolic distance ratio
approaches graph distance ratio, but the Euclidean is constant.

     d(x, x)
d(x, O) + d(y, O)

        O


   x         y


                      William L. Hamilton, McGill University, MILA, FAIR   14
Natural hierarchy in hyperbolic space
• Nodes that are closer to the
  origin are close to a larger
  number of other nodes.
• These nodes are “higher up”
  in the hierarchy.




                      William L. Hamilton, McGill University, MILA, FAIR   15
Poincare embeddings: The basic idea

1. Take your favorite node embedding
   algorithm (e.g., node2vec, DeepWalk).
2. Replace Euclidean distances with
   hyperbolic distances.



                William L. Hamilton, McGill University, MILA, FAIR   16
Poincare embeddings: The basic idea

1. Take your favorite node embedding
   algorithm (e.g., node2vec, DeepWalk).
2. Replace Euclidean distances with
          But   it’s not
   hyperbolic distances.
                         that  simple!



                William L. Hamilton, McGill University, MILA, FAIR   17
Problem: SGD is different in hyperbolic space!

Issues:
 1. We are in a bounded space.
 2. Entire curvature of space is different.

Solution: Use Riemannian SGD!




                    William L. Hamilton, McGill University, MILA, FAIR   18
Riemannian SGD
• Generalizes SGD to Riemannian manifolds.
• Core equation:



• Riemannian gradient (defined over tangent space)
• Learning rate
• Retraction from tangent space to manifold


                     William L. Hamilton, McGill University, MILA, FAIR   19
Riemannian gradient
• How to get Riemannian gradient on the Poincare ball, 𝐻% ?
• Luckily, 𝐻% is conformal to ℝ% , i.e angles are the same.
• Only difference is a scaling of the metric tensor:




 Poincare ball metric   Standard Euclidean
  tensor at point 𝑥        metric tensor



                        William L. Hamilton, McGill University, MILA, FAIR   20
Riemannian gradient
• How to get Riemannian gradient on the Poincare ball, 𝐻% ?




• Solution: Scale Euclidean gradient!

           rH L(✓) = g✓ 1 rE L(✓)

                      William L. Hamilton, McGill University, MILA, FAIR   21
Putting things together
• Use simple “natural gradient” retraction
• Project back to unit ball when necessary



• Final SGD update on 𝐻% :




                      William L. Hamilton, McGill University, MILA, FAIR   22
Experiments

• How good are hyperbolic embeddings for
  reconstruction and link prediction?
• Compare to Euclidean and TransE-style embeddings.


• Evaluate on tree-structured taxonomy and
  hierarchical social networks.

                  William L. Hamilton, McGill University, MILA, FAIR   23
Embedding taxonomies (i.e., WordNet)

• Take transitive closure of WordNet hypernym tree.
   • 82,115 nouns and 743,241 hypernymy relations
• Optimize (on a training subset):                                                            distance
                                                                                            (Euclidean,
                                                                                           hyperbolic, or
                                                                                              TransE)




        observed/training                                               negative samples
            relations
                        William L. Hamilton, McGill University, MILA, FAIR                         24
Embedding taxonomies (i.e., WordNet)



Significant
gains in low
dimensions!




               William L. Hamilton, McGill University, MILA, FAIR   25
Embedding taxonomies (i.e., WordNet)




   1) After 20 epochs                                            2) After convergence
                        William L. Hamilton, McGill University, MILA, FAIR              26
Embedding networks

• Try to predict missing edges social networks.
• Define edge-likelihood as edge probability “radius” for edge
                                                                                        consideration




distance (Euclidean, hyperbolic, or TransE)                                          Temperature or
                                                                                  steepness of logistic
• Optimize via cross-entropy and negative sampling
  (as in the taxonomy setting)
                             William L. Hamilton, McGill University, MILA, FAIR                           27
Embedding networks

• Predict missing edges social networks.
1. Take four social/citation networks
   •   AstroPh (18,772 nodes; 198,110 edges)
   •   CondMat (23,133 nodes; 93, 497 edges)
   •   GrQc (5,242 nodes; 14,496 edges)
   •   HepPh (12,208 nodes; 118,521 edges)
2. Randomly split edges into train/val/test
3. Optimize on train edges and evaluate mean average
   precision (MAP) on val/test edges.


                           William L. Hamilton, McGill University, MILA, FAIR   28
Embedding networks (results)



Significant
gains in low
dimensions!




               William L. Hamilton, McGill University, MILA, FAIR   29
Summary

• There are benefits to embedding graphs in non-
  Euclidean spaces.

• Hyperbolic spaces are good for trees and hierarchies.
  (Spherical spaces would be good for cycles).

• But optimization becomes non-trivial…


                      William L. Hamilton, McGill University, MILA, FAIR   30
Extensions and what’s next

• Nickel and Kiela’s follow-up ICML 2018 work:
   •   Use Lorenz model instead of Poincare ball.
   •   More stable, better performance.
• Hyperbolic GCNs?
• Product spaces that get benefits of all different kinds of
  geometries?




                          William L. Hamilton, McGill University, MILA, FAIR   31

