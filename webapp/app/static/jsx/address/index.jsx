import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';
import { buildQueryString, getApi } from '../components/utils';
import axios from 'axios';
import example from '../../data/example';
import QueryInfo from './QueryInfo';
import TornadoInfo from './TornadoInfo';
//import schemaResponse from '../../data/schema';
import TpoolOverall from './TpoolOverall';
import TpoolStats from './TpoolStats';
import { QueryObjContext } from '../components/Contexts';
import SortAndFilters from '../components/SortAndFilters';
import Pagination from '../components/Pagination';
import AddressSearchBar from '../components/AddressSearchBar';

import AccordionOfResults from '../components/AccordionOfResults';
import HaveIBeenCompromised from './HaveIBeenCompromised';
import ChooseTornadoPool from '../components/ChooseTornadoPool';
import exchangeData from '../../data/exchange';


//to be displayed instead of listed cluster, if no clusters were found. 
const NoClusters = (
    <div>
        <div className="center-inside">
            No clusters found.
            <br />
            <img width="200" src="/static/img/spy.png" />
            <br />
            <div>you're anonymous, harry!</div>
        </div>
    </div>
)

const AddressClusterHeader = (
    <div className="results col-12">
        <div className="panel-title">
            LINKED ADDRESSES
        </div>
    </div>
)

function ClusterPage(props) {
    const { params } = props;

    let [queryObj, setQuery] = useState({}); //sorts and filters 
    const [inputAddress, setInputAddress] = useState('');
    const [pageResults, setPageResults] = useState([]);
    const [firstView, setFirstView] = useState(true);
    const [showResultsSection, setShowResultsSection] = useState(false);
    const [loadingCluster, setLoadingCluster] = useState(false);
    const [loadingQuery, setLoadingQuery] = useState(false);
    const [queryInfo, setQueryInfo] = useState({}); //info about the input 
    const [tornado, setTornado] = useState({});
    const [aliases, setAliases] = useState({});
    const [schema, setSchema] = useState({});
    const [paginationData, setPaginationData] = useState({});
    const [searchType, setSearchType] = useState(null);

    const getAliases = () => {
        getApi('/utils/aliases', response => {
            setAliases(response.data);
        });
    }

    // initializing
    useEffect(() => {
        getAliases();

        const addr = params.get('address');
        if (addr !== null) {
            setInputAddress(addr);
            submitInputAddress(addr);
        }
    }, []);

    const setSort = attrObj => {
        getNewResults(false, { page: 0, ...attrObj });
    }

    const getNewResults = (newAddress, newQueries) => {
        if (newQueries === 'clear' || newAddress) {
            const allQueries = Object.keys(queryObj);
            for (const key of allQueries) {
                if (key.startsWith('filter_')) {
                    delete queryObj[key];
                }
            }
            setQuery(queryObj);
        } else if (newQueries) { //assume it's an object. 
            for (const [key, value] of Object.entries(newQueries)) {
                queryObj[key] = value;
            }
            setQuery(queryObj);
        }
        setShowResultsSection(true);
        const queryString = buildQueryString(queryObj);
        setLoadingCluster(true);
        if (newAddress) {
            setLoadingQuery(true);
        }

        axios.get('/search' + queryString)
            .then(function (response) {
                setLoadingQuery(false);
                setLoadingCluster(false);
                // response = schemaResponse;
                // response = example;
                // response = exchangeData;
                console.log('expecting success, data, and is_tornado', response.data, typeof(response.data));
                const { success, data, is_tornado } = JSON.parse(response.data);
                if (is_tornado === 1) {
                    const { query } = data;
                    setQueryInfo(query);
                    setSearchType('tornadoPool');
                } else if (is_tornado === 0) {
                    const { cluster, metadata, query, tornado } = data;
                    console.log('expecting cluster, metadata, query, tornado', data);
                    const { cluster_size, limit, num_pages, page } = metadata;
                    setPaginationData({ total: cluster_size, limit, num_pages, page });
                    const { sort_default } = metadata;
                    const { attribute, descending } = sort_default;
                    if (newAddress) {
                        queryObj.sort = attribute;
                        queryObj.descending = descending;
                        setQuery(queryObj);
                        setSchema(metadata.schema);
                        setQueryInfo(query);
                        setTornado(tornado);
                    }
                    setSearchType('other')

                    if (success === 1 && cluster.length > 0) {
                        setPageResults(cluster);
                    } else {
                        setPageResults([]);
                    }
                }

            })
            .catch(function (error) {
                setPageResults([]);
                setLoadingQuery(false);
                setLoadingCluster(false);
                console.log(error);
            }).finally(() => {
                if (firstView) {
                    setFirstView(false);
                }
            });
    }

    const submitInputAddress = addr => {
        //param is optional. otherwise, use the inputAddress var

        queryObj = {
            address: addr
        }; // clears all the sort, filters, page, etc.
        setQuery(queryObj);
        getNewResults(true);
        if (params.get('address') !== addr) {
            window.location.href = '/cluster?address=' + addr;
        }
        // window.history.pushState(null, null, "?address=" + addr);
    }



    return (
        <div className="container">
            <div className="row">
                <Header current={'address'} />

                <div className="col-12 halved-bar">
                    <div className="row instruct">
                        <div className="col-12">
                            {firstView && <div id="instructions">
                                Enter an ethereum address (or ENS name) to see likely connected ethereum addresses (ie. its cluster)
                                based on public data on previous transactions, or use the Tornado Cash Pool Anonymity Auditor.
                            </div>}
                            <div className="all-input-search row" >
                                <AddressSearchBar onSubmit={submitInputAddress} inputAddress={inputAddress} setInputAddress={setInputAddress} />
                                <ChooseTornadoPool />
                            </div>

                            {loadingQuery && <div id="spinner" className="center-inside">
                                <div className="spinner-border" role="status">
                                    <span className="sr-only">Loading...</span>
                                </div>
                            </div>}

                            {searchType === 'tornadoPool' &&
                                <>
                                    {showResultsSection &&
                                        <div>
                                            <div className="row results-section ">

                                                <TpoolOverall data={queryInfo} loading={loadingQuery} />
                                                {queryInfo.metadata && <TpoolStats data={queryInfo.metadata.stats} aliases={aliases} />}

                                            </div>
                                            <HaveIBeenCompromised aliases={aliases} tcashAddr={inputAddress} />
                                        </div>
                                    }

                                </>}
                            {searchType === 'other' && <>
                                {showResultsSection && <div className="row results-section">

                                    <QueryInfo data={queryInfo} loading={loadingQuery} aliases={aliases} />
                                    <TornadoInfo data={tornado} aliases={aliases} />

                                </div>}
                                {showResultsSection &&
                                    <QueryObjContext.Provider value={queryObj}>

                                        <AccordionOfResults
                                            myClassName="linked-adress"
                                            sectionHeader={AddressClusterHeader}
                                            rowTitle='address'
                                            rowBadge='entity'
                                            Pagination={<Pagination paginationData={paginationData} getNewResults={getNewResults} />}
                                            results={pageResults}
                                            loading={loadingCluster}
                                            aliases={aliases}
                                            noDataComponent={NoClusters}
                                            startIndex={paginationData.page * paginationData.limit}
                                            SortAndFilters={<SortAndFilters schema={schema} setSort={setSort} getNewResults={getNewResults} />}
                                        />
                                    </QueryObjContext.Provider>

                                }
                            </>}

                        </div >
                    </div>
                </div>
                <Footer />
            </div>
        </div>
    )
}

export default ClusterPage;