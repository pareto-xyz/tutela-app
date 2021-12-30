import React, { useEffect, useState } from 'react';
import AddressSearchBar from '../components/AddressSearchBar';
import Header from '../components/Header';
import { getApi } from '../../js/utils';
import responseExample from '../../data/txns';
import QueryInfo from '../address/QueryInfo';

function TransactionPage({ params, aliases }) {
    const [inputAddress, setInputAddress] = useState('');
    const [firstView, setFirstView] = useState(true);
    const [loadingOverall, setLoadingOverall] = useState(false);
    const [queryInfo, setQueryInfo] = useState({});

    //in case url already sets it up. 
    useEffect(() => {
        const addr = params.get('address');
        if (addr !== null) {
            setInputAddress(addr);
            loadNewData(addr);
        }
    }, []);

    const loadNewData = addr => {
        setLoadingOverall(true);
        getApi('/search/transaction?address=' + addr, response => {
            response = responseExample;
            const { data, success } = response.data;
            console.log(data);
            if (success === 1) {
                const { metadata, plotdata, query, transactions } = data;
                setQueryInfo(query);
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
            <Header current={'transactions'} />
            <div className=" col-12 halved-bar">
                {firstView &&
                    <div id="instructions">
                        Enter an ethereum address (or ENS name) to see history of transactions that reduced anonymity.
                    </div>}
                <div className="row" >
                    <AddressSearchBar
                        showTornadoHovers={false}
                        myClassName="col-12"
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
                </div>}
            </div>



        </div>
    )
}

export default TransactionPage;