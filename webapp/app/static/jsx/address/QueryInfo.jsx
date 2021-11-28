import React from 'react';
import AgnosticTable from '../components/AgnosticTable';

const TO_IGNORE = new Set(['metadata', 'address', 'id', 'anonymity_score']);

export default function QueryInfo({ data, loading, aliases }) {
    let { metadata, anonymity_score } = data;
    if (anonymity_score === undefined) {
        anonymity_score = 1;
    }
    const combined = { ...data, ...metadata };

    return (
        <div className="query-info ">
            <div className="panel-sub">about your input</div>
            <div className="panel-title">
                OVERALL INFO
            </div>


            {anonymity_score >= 0 && <div className="anon-score-group">
                anonymity score: &nbsp;{anonymity_score * 100} &nbsp;/ 100
                <div data-container="body" data-toggle="popover" data-placement="bottom" data-trigger="hover"
                    data-content="The higher the anonymity score, the less we believe this address or transaction has revealed about its privacy. Number of reveals, the connectedness of addresses and the types of reveal affect this."
                    className="help-circle">?</div>
            </div>}
            {loading && <div id="spinner" className="justify-content-center">
                <div className="spinner-border" role="status">
                    <span className="sr-only">Loading...</span>
                </div>
            </div>}
            <AgnosticTable toIgnore={TO_IGNORE} keyValues={Object.entries(combined)} />
        </div>
    );
}