import axios from 'axios';
import React, { useState, useRef } from 'react';
import { InputGroup, FormControl, Form, } from 'react-bootstrap';
//import { compromisedTcashAddrResponse, failedCompromisedTcashResponse } from '../../data/ihavebeencompromised';
import ListOfResults from '../components/ListOfResults';


export default function HaveIBeenCompromised({ tcashAddr, aliases }) {
    const inputEl = useRef(null);
    const [val, setVal] = useState('');
    const [numCompromised, setNumCompromised] = useState(null);
    const [compromisedTxns, setCompromisedTxns] = useState(null);
    const [invalid, setInvalid] = useState(false);

    const checkIfCompromised = address => {
        axios.get('/search/compromised?address=' + address + '&pool=' + tcashAddr).then(response => {
            //response = compromisedTcashAddrResponse;
            const { success, data } = response.data;
            if (!success) {
                setNumCompromised(null);
                setCompromisedTxns(null);
                setInvalid(true);
                return;
            }
            const { compromised_size, compromised } = data;
            setNumCompromised(compromised_size);
            setCompromisedTxns(compromised);
            setInvalid(false);
        })
    }

    return (
        <div className="row">
            <div className="col-12">
                <div className="query-info">
                    <div className="row">
                        <div className="col-12 panel-sub">
                            check if your transactions have been compromised
                        </div>
                        <div className="col-12 panel-title">
                            YOUR COMPROMISED TXNS IN THIS POOL
                        </div>
                        <InputGroup className="col-12">
                            <FormControl className="rounded specific-result"
                                placeholder="enter a deposit or withdrawal address used with this pool to check for compromised txs"
                                onChange={(e) => {
                                    e.preventDefault();
                                    setVal(e.target.value);
                                }}
                                onKeyPress={(e) => {
                                    if (e.key !== 'Enter') {
                                        return;
                                    }
                                    checkIfCompromised(val);
                                }}
                                ref={inputEl}
                                isInvalid={invalid}
                            />
                            <Form.Control.Feedback type='invalid'>Please input a valid deposit address.</Form.Control.Feedback>

                        </InputGroup>


                        {compromisedTxns && <ListOfResults
                            sectionHeader={numCompromised !== null && <div className="results-section col-12">Total deposits compromised: {numCompromised}</div>}
                            rowTitle='transaction'
                            rowBadge='heuristic'
                            results={compromisedTxns}
                            aliases={aliases}
                        />}
                    </div>
                </div>
            </div>
        </div>
    )
}