import React, { useState }  from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';
import AddressSearchBar from '../components/AddressSearchBar';


export default function IndexPage() {

    const [inputAddress, setInputAddress] = useState('');

    const onAddressSubmit = (addr) => {
        window.history.pushState(null, null, "?address=" + addr);
    }

    return (
        <>
            <Header />
            <div className="full-page container main">
                <h1>assess your anonymity.</h1>
                <AddressSearchBar onSubmit={onAddressSubmit} inputAddress={inputAddress} setInputAddress={setInputAddress}/>
                <div className="main-desc">Enter an ethereum address or ENS name to see likely connected ethereum addresses (ie. its cluster)
                    based on public data on previous transactions. </div>
                <div className="results">
                    <table id="results-table">
                    </table>
                </div>
            </div>
            <div className="full-page container" id="about">
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
                    We use data science and Ethereum and Tornado Cash reveals to <strong>probablistically estimate which
                        Ethereum addresses are affiliated with single entities. </strong>


                </div>
                <div className="section-desc">
                    We built this application as part of a <a
                        href="https://torn.community/t/funded-bounty-anonymity-research-tools/1437">Tornado Cash Grant</a>.
                    We respect your privacy and so do not save search results.
                </div>
            </div>
            <div className="full-page container" id="reveals">
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
                    <div className="center-blocks">
                        <div>
                            <div className="block-title">
                                Deposit Address Reuse
                            </div>
                            <div className="block-text">
                                When you send tokens to your account at a centralized exchange, the exchange creates a unique
                                deposit addresses for each customer. If you send tokens from multiple ethereum wallets, they
                                will likely send to the same deposit address, linking your two wallets.
                            </div>
                        </div>
                        <div>
                            <div className="block-title">
                                Time-of-Day
                                Transaction Activity
                            </div>
                            <div className="block-text">
                                Ethereum transaction timestamps reveal the daily activity patterns of the account owner.
                                Consistent time-of-day activity across multiple addresses can be used to link multiple wallets.
                            </div>
                        </div>

                        <div>
                            <div className="block-title">
                                Gas Price
                                Distribution
                            </div>
                            <div className="block-text">
                                Ethereum transactions also contain the gas price, which is
                                usually automatically set by wallet softwares. Users rarely
                                change this setting manually, so may reveal when two addresses use the same version of a wallet
                                (e.g. v[xx] of MetaMask).
                            </div>
                        </div>
                    </div>
                </div>
                <br />
                <br />
                <div className="subsect">
                    <h4>tornado cash heuristics</h4>
                    <div className="center-blocks">
                        <div>
                            <div className="block-title">
                                Synchronous
                                Tx Reveal
                            </div>
                            <div className="block-text">
                                If you consistently make txs with a “source” wallet and a “destination” wallet at roughly
                                similar times, you may reveal the two wallets have the same owner.
                            </div>
                        </div>
                        <div>
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
                        <div>
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
        </>
    );
}