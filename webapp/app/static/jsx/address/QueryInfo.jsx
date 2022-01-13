import React from 'react';
import AgnosticTable from '../components/AgnosticTable';
import MyTooltip from '../components/MyTooltip';

const TO_IGNORE = new Set(['metadata', 'id', 'anonymity_score', 'start_date', 'end_date']);

export default function QueryInfo({ data, loading=false, aliases }) {
    let { metadata, anonymity_score } = data;
    if (anonymity_score === undefined) {
        anonymity_score = 1;
    }
    const combined = { ...data, ...metadata };

    const displayedScore = (anonymity_score * 100).toFixed(0);

    return (
        <div className="col-md-12 col-lg-6">
            <div className="query-info ">
                <div className="panel-sub">about your input</div>
                <div className="panel-title">
                    OVERALL INFO
                </div>

                {anonymity_score >= 0 && <div className="anon-score-group">
                    anonymity score: &nbsp;{displayedScore} &nbsp;/ 100
                    <MyTooltip tooltipText={'The higher the anonymity score, the less we believe this address or transaction has revealed about its privacy. Number of reveals, the connectedness of addresses and the types of reveal affect this.'} />
                </div>}
                
                {!loading && <AgnosticTable aliases={aliases} toIgnore={TO_IGNORE} keyValues={combined} />}
            </div>
        </div>
    );
}