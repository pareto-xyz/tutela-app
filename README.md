# Tutela: an Ethereum and Tornado Cash Anonymity Tool

The repo contains open-source code for [Tutela](http://tutela.xyz), an anonymity tool for Ethereum and Tornado Cash users. For a more technical description, please refer to the public [whitepaper](https://arxiv.org/abs/2201.06811). 

## About Tutela

In response to the [Tornado Cash (TC) Anonymity Research Tools Grant](https://torn.community/t/funded-bounty-anonymity-research-tools/1437), we have built [Tutela](http://tutela.xyz), an Ethereum wallet anonymity detection tool, to tell you if your blockchain transactions have revealed anything about your identity. *What does this mean?* Well, for example, if you have used multiple Ethereum wallets to send tokens to a single centralized exchange deposit address, you may have revealed that your wallets are owned by the same entity.

We'd love to get user feedback! Tell us what you like, what you don’t and what you think is missing! Please leave your feedback in the *Tutela-Product-Feedback* channel of the [Tornado Cash Discord](https://discord.gg/xGDKUbMx).

### The Tornado Cash User's Dilemma

Tornado cash users have multiple addresses and use Tornado Cash to hide this fact. We believe the most important need for this user base is to know whether their addresses can already be connected by third parties. Conversely, for Tornado Cash, compromised transactions could reduce the size of the anonymity set for each token pool.

### Tutela, an Anonymity Detection Tool

In response, Tutela has focused on informing users which of their Ethereum addresses are "affiliated" (a non-blockchain analogy would be [haveibeenpwned.com](https://haveibeenpwned.com)) by parsing the Ethereum graph of transaction. This involves two Ethereum-wide heuristics and five Tornado Cash -specific heuristics (i.e. reveals) that investigate transactions in and out of Tornado Cash pools. 

### Ethereum Heuristics

Across all of Ethereum, we would like to cluster together addresses that likely belong to the same entity. 

#### [Deposit Address Reuse](https://arxiv.org/pdf/2005.14051.pdf)

When you send tokens from an Ethereum wallet to your account at a centralized exchange, the exchange creates a unique deposit address for each customer. If you reuse the same deposit address by sending tokens from multiple Ethereum wallets to it, your two wallets can be linked. Even if you send tokens from multiple wallets to multiple deposits, all of these addresses can be linked. In this way, it is possible to build a complex graph of address relationships.

#### [Diff2Vec](https://arxiv.org/abs/2001.07463)

Every Ethereum address is mapped to a point in a high dimensional vector space using a machine Learning algorithm. For every Ethereum entity, the goal of the vector embedding is to summarize which addresses this entity interacts with the most. This is done by creating a large Ethereum graph where nodes represent addresses and edges represent transactions -- for every node, a local subgraph is created through random walks, which is then featurized and put through Word2Vec. 

### Tornado Cash Heuristics

If we focus on users interacting with Tornado Cash pools, we can apply different heuristics to link together deposit and withdraw transactions. 

#### Address Match

If a deposit transaction and a withdrawal transaction to a specific Tornado Cash pool share the same wallet address, then we assume the address is compromised (e.g. they may be a yield miner who does not care about anonymity), and should not add to the anonymity of future Tornado Cash transactions for that pool.

#### Unique Gas Price 

Pre EIP-1559 Ethereum transactions contained a gas price. Users can set their wallet gas fee and pay a very specific gas fee (e.g. 147.4535436 Gwei) when they deposit in a Tornado Cash pool. If they also withdraw from that same Tornado cash pool, using the same wallet application (e.g. Metamask), but a different wallet address and haven’t changed the gas fee, it could reveal that two addresses are connected.

#### Multi-Denomination Match

If we observe a single address depositing multiple times to several pools and then a second address withdrawing the exact same amount from the same pools, it is likely that the two addresses belong to the same entity. For example, if Alice deposits 5 times to the 1 ETH pool, 3 times to the 100 DAI pool, and 4 times to the 0.1 ETH pool, and Bob withdraws identically, then Alice and Bob potentially are the same individual. 

#### Linked Address Match

If address A deposits to a Tornado Cash pool and address B withdraws from the same pool but we observe frequent interactions between Address A and B outside of Tornado Cash (more general Ethereum transactions), it potentially indicates that address A and B are owned by the same entity. 

#### Careless Anonymity Mining 

Anonymity mining is a clever way to incentivize users to participate in mixing. However, if users carelessly claim their Anonymity Points (AP) or Tornado tokens, then they can reduce their anonymity set. For instance, if a user withdraws their earned AP tokens to a deposit address, then we can approximate the maximum time a user has left their funds in the mixing pool. This is because users can only claim AP and TORN tokens after deposit transactions that were already withdrawn.

### We Need Your Help!

Tutela is still in its very early stages and we are looking for feedback at all levels. Let us know your thoughts, critiques, and suggestions in the *Tutela-Product-Feedback* channel of the [Tornado Cash Discord](https://discord.gg/xGDKUbMx).. How can we make Tutela something useful for you? What features or heuristics are we missing?

### Technical Summary

Ethereum and Tornado Cash transactions are downloaded using BigQuery. The deposit address reuse algorithm was adapted from the existing implementation in [etherclust](https://github.com/etherclust/etherclust). Our Python implementation can be found in `src/`; it is written to scalably operate over the >1 Tb of Ethereum data. The Tornado-specific heuristics can be found in `scripts/tornadocash`, again written in Python. The Tutela web application lives in `webapp/` and is written in Flask with a PostgreSQL database for storing clusters. The frontend is written in Javascript, HTML, and CSS. 

## Updates

We aim to provide consistent updates over time as we improve Tutela. 

- **(11/17)** We posted a pre-beta version of Tutela to the Tornado Cash community for feedback.
- **(11/23)** We open-sourced the Tutela implementation and will make all future improvements public through pull requests. Since 11/17, we increased the number of Centralized Exchange Addresses used in clustering from 171 to 332, and added a list of well-known addresses that we omit from consideration when classifying deposits. Improvements were made to the Tcash gas price heuristic and we have added the Tcash synchronous Tx reveal: searching by address will now return TCash specific information for all addresses. Several bugfixes were implemented, such as address casing, incorrect deposit names, deposit reuse hyperparameters.
- **(12/22)** We added five Tornado Cash heuristics. If you search an address who has used Tornado Cash, Tutela will now show compromised transactions. If you search an address corresponding to a Tornado Cash pool, you will get statistics on the pool's true anonymity size. We are currently in progress on deploying Diff2Vec at scale. 
- **(12/28)** Deployed first instance if Diff2Vec. Some improvements to be made.
- **(1/10)** Working on a live updating pipeline.
- **(1/19)** Completed live updating pipeline and another round of changes from user-feedback.

## Contributors

Development of the web application and clustering was done by [mhw32](https://github.com/mhw32), [kkailiwang](https://github.com/kkailiwang), [Tiggy560](https://github.com/Tiggy560), and [nickbax](https://github.com/nickbax), with support from [Convex Labs](https://www.convexlabs.xyz). Development of TCash heuristics was done by [seresistvanandras](https://github.com/seresistvanandras), [unbalancedparentheses](https://github.com/unbalancedparentheses), [tomasdema](https://github.com/tomasdema), [entropidelic](https://github.com/entropidelic), [HermanObst](https://github.com/HermanObst), and [pefontana](https://github.com/pefontana). 
