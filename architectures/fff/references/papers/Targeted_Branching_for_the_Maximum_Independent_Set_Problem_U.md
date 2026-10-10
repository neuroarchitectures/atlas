# Targeted Branching for the Maximum Independent Set Problem Using Graph Neural Networks

> Source: `https://doi.org/https://doi.org/10.4230/lipics.sea.2024.20`

---

**Authors:** Silva, Gabriel, Rodrigues, Mário, Teixeira, António, Amorim, Marlene

**Published:** 2024

**Concepts:** Computer science, Node (physics), Graph, Feature learning, Representation (politics)

## Abstract

Identifying a maximum independent set is a fundamental NP-hard problem. This problem has several real-world applications and requires finding the largest possible set of vertices not adjacent to each other in an undirected graph. Over the past few years, branch-and-bound and branch-and-reduce algorithms have emerged as some of the most effective methods for solving the problem exactly. Specifically, the branch-and-reduce approach, which combines branch-and-bound principles with reduction rules, has proven particularly successful in tackling previously unmanageable real-world instances. This progress was largely made possible by the development of more effective reduction rules. Nevertheless, other key components that can impact the efficiency of these algorithms have not received the same level of interest. Among these is the branching strategy, which determines which vertex to branch on next. Until recently, the most widely used strategy was to choose the vertex of the highest degree. In this work, we present a graph neural network approach for selecting the next branching vertex. The intricate nature of current branch-and-bound solvers makes supervised and reinforcement learning difficult. Therefore, we use a population-based genetic algorithm to evolve the model’s parameters instead. Our proposed approach results in a speedup on 73% of the benchmark instances with a median speedup of 24%.

**DOI:** https://doi.org/https://doi.org/10.4230/lipics.sea.2024.20
