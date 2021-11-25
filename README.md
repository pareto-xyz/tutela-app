# Tutela: an Ethereum and Tornado Cash Anonymity Tool

The repo contains open-source code for [Tutela](http://tutela.xyz), an anonymity tool for Ethereum and Tornado Cash users. 

## About Tutela

In response to the [Tornado Cash (TC) Anonymity Research Tools Grant](https://torn.community/t/funded-bounty-anonymity-research-tools/1437), we have built [Tutela v1](http://tutela.xyz), an Ethereum wallet anonymity detection tool, to tell you if your blockchain transactions have revealed anything about your identity. *What does this mean?* Well, for example, if you have used multiple Ethereum wallets to send tokens to a single centralized exchange deposit address, you may have revealed that your wallets are owned by the same entity.

We'd love to get user feedback! Tell us what you like, what you don’t and what you think is missing! Please leave your feedback in the *Tutela-Product-Feedback* channel of the [Tornado Cash Discord](https://discord.gg/xGDKUbMx).

### The Tornado Cash User's Dilemma

Tornado cash users have multiple addresses and use Tornado Cash to hide this fact. We believe the most important need for this user base is to know whether their addresses can already be connected by third parties.

### Tutela, an Anonymity Detection Tool

In response, our initial MVP has focused on informing users which of their Ethereum addresses are "affiliated" (a non-blockchain analogy would be [haveibeenpwned.com](https://haveibeenpwned.com)). This involves using a clustering algorithm and three heuristics (i.e. reveals) so far, the [Ethereum deposit address reuse heuristic](https://link.springer.com/chapter/10.1007/978-3-030-51280-4_33), the [Tornado Cash unique gas price heuristic](https://arxiv.org/abs/2005.14051) and the Tornado Cash Synchronous transaction heuristic. We plan to refine and add additional heuristics over time.

### Current Heuristics

#### Ethereum Deposit Address Reuse Heuristic

When you send tokens from an Ethereum wallet to your account at a centralized exchange, the exchange creates a unique deposit address for each customer. If you reuse the same deposit address by sending tokens from multiple Ethereum wallets to it, your two wallets can be linked. Even if you send tokens from multiple wallets to multiple deposits, all of these addresses can be linked. In this way, it is possible to build a complex graph of address relationships.

#### Tornado Cash Pools Unique Gas Price Heuristic

Pre EIP-1559 Ethereum transactions contained a gas price. Users can set their wallet gas fee and pay a very specific gas fee (e.g. 147.4535436 Gwei) when they deposit in a Tornado Cash pool. If they also withdraw from that same Tornado cash pool, using the same wallet application (e.g. Metamask), but a different wallet address and haven’t changed the gas fee, it could reveal that two addresses are connected.

#### Tornado Cash Pools Synchronous Tx Heuristic

If a deposit transaction and a withdrawal transaction to a specific Tornado Cash pool share the same wallet address, then we assume the address is compromised (e.g. they may be a yield miner who does not care about anonymity), and should not add to the anonymity of future Tornado Cash transactions for that pool.

### We Need Your Help!

Tutela is still in its very early stages and we are looking for feedback at all levels. Let us know your thoughts, critiques, and suggestions in the *Tutela-Product-Feedback* channel of the [Tornado Cash Discord](https://discord.gg/xGDKUbMx).. How can we make Tutela something useful for you? What features or heuristics are we missing?

### Next Steps

Our plan for the next two months is to refine and develop Tutela v1 by:

1. Getting your feedback!
2. Refining the deposit reuse heuristic
3. Adding anonymity set scoring for Tornado Cash pools
4. Providing transaction by transaction reveal data (studying anonymity over time)
5. Identifying, testing and implementing Tornado Cash Specific Heuristics:
    1. **Linking equal value deposits and withdrawals to specific deposit and withdrawal addresses** - if there are multiple (say 12) deposit transactions coming from a deposit address and later there are 12 withdraw transactions to the same withdraw address, then we could link all these deposit transactions to the withdraw transactions
    2. **Careless TC anonymity mining** - anonymity mining is a clever way to incentivize users to participate in mixing. However, if users carelessly claim their Anonymity Points (AP) or Tornado tokens, then they can reduce their anonymity set. For instance, if a user withdraws their earned AP tokens to a deposit address, then we can approximate the maximum time a user has left their funds in the mixing pool. This is because users can only claim AP and TORN tokens after deposit transactions that were already withdrawn.
    3. **Profiling deposit and withdrawal addresses** - collect and analyze the behaviour of all addresses that have interacted with Tornado cash pools
    4. **Wallet fingerprinting** - different wallets work in different ways. We have several ideas on how we can distinguish between them. It will allow us to further fragment the anonymity sets of withdraw transactions.

### Technical Summary

Ethereum and Tornado Cash transactions are downloaded using BigQuery. The deposit address reuse algorithm was adapted from the existing implementation in [etherclust](https://github.com/etherclust/etherclust). Our Python implementation can be found in `src/`; it is written to scalably operate over the >1 Tb of Ethereum data. The Tornado-specific heuristics can be found in `scripts/tornadocash`, again written in Python. The Tutela web application lives in `webapp/` and is written in Flask with a PostgreSQL database for storing clusters. The frontend is written in Javascript, HTML, and CSS. 

## Updates

We aim to provide consistent updates over time as we improve Tutela. 

- **(11/17)** We posted a pre-beta version of Tutela to the Tornado Cash community for feedback.
- **(11/23)** We open-sourced the Tutela implementation and will make all future improvements public through pull requests. Since 11/17, we increased the number of Centralized Exchange Addresses used in clustering from 171 to 332, and added a list of well-known addresses that we omit from consideration when classifying deposits. Improvements were made to the Tcash gas price heuristic and we have added the Tcash synchronous Tx reveal: searching by address will now return TCash specific information for all addresses. Several bugfixes were implemented, such as address casing, incorrect deposit names, deposit reuse hyperparameters.

## Contributors

Development of the web application and clustering was done by [mhw32](https://github.com/mhw32), [kkailiwang](https://github.com/kkailiwang), [Tiggy560](https://github.com/Tiggy560), and [nickbax](https://github.com/nickbax), with support from [Convex Labs](https://www.convexlabs.xyz). Development of TCash heuristics was done by [seresistvanandras](https://github.com/seresistvanandras), [unbalancedparentheses](https://github.com/unbalancedparentheses), [tomasdema](https://github.com/tomasdema), [entropidelic](https://github.com/entropidelic), [HermanObst](https://github.com/HermanObst), and [pefontana](https://github.com/pefontana). 
