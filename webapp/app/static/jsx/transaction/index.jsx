import React, { useEffect, useState } from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';
import AddressSearchBar from '../components/AddressSearchBar';
import { buildQueryString, getApi } from '../../js/utils';
//import responseExample from '../../data/txns';
import QueryInfo from '../address/QueryInfo';
import RevealTimeline from './RevealsTimeline';
import AccordionOfResults from '../components/AccordionOfResults';

const TransactionsListHeader = ({ start_date, end_date }) => (
    <div className="results col-12">
        <div className="panel-title">
            REVEALING TRANSACTIONS ({start_date} - {end_date})
        </div>
    </div>
)

const NoReveals = (
    <div>
        <div className="center-inside">
            You have not made any reveals! .
            <br />
            <img width="200" src="/static/img/spy.png" />
            <br />
            <div>nice.</div>
        </div>
    </div>
)

function TransactionPage({ params, aliases }) {
    const [inputAddress, setInputAddress] = useState('');
    const [firstView, setFirstView] = useState(true);
    const [loadingOverall, setLoadingOverall] = useState(false);
    const [queryInfo, setQueryInfo] = useState({});
    const [transactions, setTransactions] = useState([]);

    //in case url already sets it up. 
    useEffect(() => {
        const addr = params.get('address');
        if (addr !== null) {
            setInputAddress(addr);
            loadNewData({ address: addr });
        }
    }, []);

    const loadNewData = (queryObj) => {

        setLoadingOverall(true);
        getApi('/search/transaction' + buildQueryString(queryObj), response => {
            //response = responseExample;
            const { data, success } = response.data;
            if (success === 1) {
                const { metadata, query, transactions } = data;
                setQueryInfo(query);
                setTransactions(transactions);
            }
        }, () => { //this is the finally.
            if (firstView) {
                setFirstView(false);
            };
            setLoadingOverall(false);
        })
    }

    const submitInputAddress = addr => {
        if (params.get('address') !== addr) {
            window.location.href = '/transactions?address=' + addr;
        }
        //otherwise, it's the same addr as before. so do nothing. 
    }

    return (
        <div className="container">
            <div className="row">
                <Header current={'transactions'} />
                <div className="col-12 halved-bar">
                    {firstView &&
                        <div id="instructions" className="instructions-transactions">
                            Enter an ethereum address (or ENS name) to see history of transactions that reduced anonymity.
                        </div>}
                    <div className="justify-center row" >
                        <AddressSearchBar
                            showTornadoHovers={false}
                            onSubmit={submitInputAddress}
                            inputAddress={inputAddress}
                            setInputAddress={setInputAddress} />
                    </div>

                    {loadingOverall && <div id="spinner" className="center-inside">
                        <div className="spinner-border" role="status">
                            <span className="sr-only">Loading...</span>
                        </div>
                    </div>}

                    {!firstView && <div className="row results-section">
                        <QueryInfo data={queryInfo} aliases={aliases} />
                        <RevealTimeline
                            loadNewData={loadNewData}
                            addr={params.get('address')}
                            aliases={aliases} />
                    </div>}
                    {!firstView && <AccordionOfResults
                        myClassName="linked-address"
                        sectionHeader={TransactionsListHeader(queryInfo)}
                        rowTitle='transaction'
                        rowBadge='heuristic'
                        results={transactions}
                        aliases={aliases}
                        noDataComponent={NoReveals}
                    />}

                </div>
                <Footer />
            </div>
        </div>
    )
}

export default TransactionPage;