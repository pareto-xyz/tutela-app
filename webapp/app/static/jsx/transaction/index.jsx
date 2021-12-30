import React, { useState } from 'react';
import AddressSearchBar from '../components/AddressSearchBar';
import Header from '../components/Header';
import { getApi } from '../components/utils';

function TransactionPage({ params }) {
    const [inputAddress, setInputAddress] = useState('');
    const [firstView, setFirstView] = useState(true);
    const [loadingOverall, setLoadingOverall] = useState(false);

    const submitInputAddress = addr => {
        setLoadingOverall(true);
        getApi('/search/transaction?address=' + addr, response => {
            response = response.data;
            console.log(response);
        }, () => { //this is the finally.
            if (firstView) {
                setFirstView(false);
            };
            setLoadingOverall(false);
        })
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
            </div>
            <div className="row">

            </div>


        </div>
    )
}

export default TransactionPage;