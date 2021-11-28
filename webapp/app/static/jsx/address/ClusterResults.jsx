import React from 'react';
import SortAndFilters from '../components/SortAndFilters';
import Pagination from '../components/Pagination';

function NoClusters() {
    return (
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
}

export default function ClusterResults(props) {
    const { results, loading } = props;
    const noResults = results.length == 0;

    return (
        <div >
            <SortAndFilters />
           {!noResults && <Pagination />} 


            <div className="results-section">
                <div className="results">
                    <div className="panel-title spaced">
                        LINKED ADDRESSES
                    </div>
                    {loading ? <div id="spinner" className="justify-content-center">
                        <div className="spinner-border" role="status">
                            <span className="sr-only">Loading...</span>
                        </div>
                    </div> : (noResults ? <NoClusters /> : <table id="results-table">
                    </table>)

                    }

                </div>
                {/* <div className="detail-page">
                    <div className="result-identifier panel-sub">
                        result&nbsp;<span id="result-number">0</span>&nbsp;out of&nbsp;<span
                            className="total-results"></span>
                    </div>
                    <div id="detail-address" className="panel-title">
                        SELECTED LINKED ADDRESS
                    </div>
                    <div id="detail-table">
                    </div>
                </div> */}
            </div>
        </div>
    )
}