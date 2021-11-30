import React from 'react';
import AgnosticTable from '../components/AgnosticTable';
import { Tooltip, OverlayTrigger } from 'react-bootstrap';

const TO_IGNORE = new Set(['metadata', 'address', 'id', 'anonymity_score']);

export default function QueryInfo({ data, loading, aliases }) {
    let { metadata, anonymity_score } = data;
    if (anonymity_score === undefined) {
        anonymity_score = 1;
    }
    const combined = { ...data, ...metadata };

    const renderHelpTooltip = props => {
        return (
            <Tooltip {...props} className="tooltip">
                The higher the anonymity score, the less we believe this address or transaction has revealed about its privacy. Number of reveals, the connectedness of addresses and the types of reveal affect this.
            </Tooltip>
        );
    }

    return (
        <div className="query-info ">
            <div className="panel-sub">about your input</div>
            <div className="panel-title">
                OVERALL INFO
            </div>


            {anonymity_score >= 0 && <div className="anon-score-group">
                anonymity score: &nbsp;{anonymity_score * 100} &nbsp;/ 100
                <OverlayTrigger
                    placement="right"
                    delay={{ show: 250, hide: 400 }}
                    overlay={renderHelpTooltip}
                >
                    <div className="help-circle">?</div>
                </OverlayTrigger>
            </div>}
            {loading && <div id="spinner" className="justify-content-center">
                <div className="spinner-border" role="status">
                    <span className="sr-only">Loading...</span>
                </div>
            </div>}
            {!loading && <AgnosticTable aliases={aliases} toIgnore={TO_IGNORE} keyValues={Object.entries(combined)} />}
        </div>
    );
}