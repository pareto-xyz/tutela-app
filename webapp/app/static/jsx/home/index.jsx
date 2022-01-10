import React, { useState } from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';
import AddressSearchBar from '../components/AddressSearchBar';
import ChooseTornadoPool from '../components/ChooseTornadoPool';


export default function IndexPage() {

    const [inputAddress, setInputAddress] = useState('');

    const onAddressSubmit = (addr) => {
        window.location.href = "/cluster?address=" + addr;
    }

    return (
        <div className="container">
            <div className="row">
                <Header />
                <div className="full-page main col-12">
                    <div className="row">
                        <h1 className="col-12">assess your anonymity.</h1>
                    </div>
                    <div className="justify-center row">
                        <AddressSearchBar onSubmit={onAddressSubmit} inputAddress={inputAddress} setInputAddress={setInputAddress} />
                    </div>
                    <div className="row instruct">
                        <div className="col-8 text-center " id="instructions" >Enter an ethereum address or ENS name to see likely connected ethereum addresses (ie. its cluster)
                            based on public data on previous transactions. </div>
                    </div>
                    <div className="row instruct instructions">
                        -- or --
                    </div>
                    <div className="row instruct instructions">
                        Use the Tornado Cash Pool Anonymity Auditor
                    </div>
                    <div className="row justify-center">
                        <ChooseTornadoPool />
                    </div>
                </div>
                <div className="full-page col-12" id="about">
                    <h3 className="left-section-header">
                        about
                    </h3>
                    <div className="section-desc">
                        Every transaction you make on the blockchain reveals more about your personal data. For individuals who care
                        to protect their privacy, they must be mindful of what transactions they make and how they make them.
                    </div>
                    <div className="section-desc ">
                        <strong>Tutela is Latin for Protection. We named this application Tutela because it
                            aims to
                            help Ethereum and Tornado Cash users protect their privacy, and understand how much they have revealed
                            about themselves through their blockchain activity.</strong>
                    </div>
                    <div className="section-desc">
                        We use data science and Ethereum and Tornado Cash reveals to <strong>probablistically estimate which Ethereum
                            addresses are affiliated with single entities and which Tornado Cash deposits may be compromised.</strong>


                    </div>
                    <div className="section-desc">
                        We built this application as part of a <a
                            href="https://torn.community/t/funded-bounty-anonymity-research-tools/1437">Tornado Cash Grant</a>.
                        We respect your privacy and so do not save search results.
                    </div>
                </div>
                <div className="full-page col-12" id="reveals">
                    <h3 className="right-section-header">
                        reveals and heuristics
                    </h3>
                    <div className="right section-desc">
                        There are lots of reveals that you can commit on the blockchain that can link your various Ethereum wallets.
                        We outline some of the key reveals below.
                    </div>
                    <br />
                    <div className="subsect">
                        <h4>ethereum heuristics</h4>
                        <div className="row center-blocks">
                            <div className="col-sm-12 col-md-6">
                                <div className="block-title">
                                    Deposit Address Reuse
                                </div>
                                <div className="block-text">
                                    When you send tokens to your account at a centralized exchange, the exchange creates a unique
                                    deposit addresses for each customer. If you send tokens from multiple ethereum wallets, they
                                    will likely send to the same deposit address, linking your two wallets.
                                </div>
                            </div>
                            <div className="col-sm-12 col-md-6">
                                <div className="block-title">
                                    Diff2Vec Machine Learning
                                </div>
                                <div className="block-text">
                                    Diff2Vec is a machine learning algorithm. Applying it to Ethereum transactions allows the
                                    clustering of potentially related addresses.
                                </div>
                            </div>


                        </div>
                    </div>
                    <br />
                    <br />
                    <div className="subsect">
                        <h4>tornado cash heuristics</h4>
                        <div className="row center-blocks">
                            <div className="col-sm-12 col-md-6 col-lg-4">
                                <div className="block-title">
                                    Address Match Reveal
                                </div>
                                <div className="block-text">
                                    Suppose the address making a deposit transaction to a Tornado Cash pool matches
                                    the address making a withdrawal transaction (from the same pool). In that case,
                                    the two transactions can be linked, and the corresponding deposit may be compromised
                                    as the user identity may be revealed. These may be TORN yield farmers who deposit and withdraw to the same address.
                                </div>
                            </div>
                            <div className="col-sm-12 col-md-6 col-lg-4">
                                <div className="block-title">
                                    Unique Gas Price Reveal
                                </div>
                                <div className="block-text">
                                    Prior to EIP-1559, Ethereum users could specify the gas price when making a
                                    deposit or withdrawal to a Tornado Cash pool. Those who do so tend to specify
                                    gas prices that are identical for deposit and withdrawal transactions, linking the
                                    wallets that made the deposit and withdrawal transactions.
                                </div>
                            </div>
                            <div className="col-sm-12 col-md-6 col-lg-4">
                                <div className="block-title">
                                    Multi-Denomination Reveal
                                </div>
                                <div className="block-text">
                                    If your “source” wallet mixes a specific set of denominations and your “destination” wallet
                                    withdraws them all (example: if you mix 3x 10 ETH, 2x 1 ETH, 1x 0.1 ETH in order to get 32.1 ETH
                                    to begin staking on the beacon chain), then you could reveal yourself within the Tornado
                                    protocol
                                    if no other wallet has mixed this exact denomination set.
                                </div>
                            </div>
                            <div className="col-sm-12 col-md-6 col-lg-4">
                                <div className="block-title">
                                    Linked Address Reveal
                                </div>
                                <div className="block-text">
                                    Suppose two addresses deposit and withdraw to the same Tornado Cash Pool. If these addresses
                                    interact outside of the Tornado Protocol, then they may be linked and their deposits compromised.
                                </div>
                            </div>
                            <div className="col-sm-12 col-md-6 col-lg-4">
                                <div className="block-title">
                                    TORN Mining
                                    Reveal
                                </div>
                                <div className="block-text">
                                    If you liquidity mine anonymity points (AP) and claim them all for TORN in a single tx with
                                    either the “source” or “destination” wallet, you will reveal exactly how long you mixed your
                                    tokens. This will perfectly connect the two addresses.
                                </div>
                            </div>

                        </div>
                    </div>
                    <Footer />
                </div>
            </div>
        </div>
    );
}