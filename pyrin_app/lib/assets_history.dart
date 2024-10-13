import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:pyrin_app/components/tokens_list.dart';
import 'package:pyrin_app/ui.dart';
import 'package:pyrin_app/core/token.dart';

class AssetsHistory extends StatelessWidget {
  final Token token;

  const AssetsHistory({Key? key, required this.token}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: PyrinColors.BACKGROUND_COLOR,
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(context),
            _buildTokenInfo(),
            _buildActionButtons(),
            _buildHistoryTabs(),
            Expanded(child: _buildHistoryList()),
            _buildGoToMarketButton(),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Row(
        children: [
          IconButton(
            icon: const Icon(Icons.arrow_back, color: PyrinColors.TEXT_COLOR),
            onPressed: () => Navigator.of(context).pop(),
          ),
          const Spacer(),
          Text(
            token.symbol,
            style: const TextStyle(
              color: PyrinColors.TEXT_COLOR,
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
          const Spacer(),
          const SizedBox(width: 48),
        ],
      ),
    );
  }

  Widget _buildTokenInfo() {
    return Column(
      children: [
        TokenIcon(symbol: token.symbol, size: 64),
        const SizedBox(height: 16),
        Text(
          '${(token.amount / 1e8).toStringAsFixed(0)} ${token.symbol}',
          style: const TextStyle(
            color: PyrinColors.TEXT_COLOR,
            fontSize: 24,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 8),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              '\$100',
              style: TextStyle(
                color: PyrinColors.TEXT_COLOR.withOpacity(0.7),
                fontSize: 16,
              ),
            ),
            const SizedBox(width: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: PyrinColors.GREEN_COLOR.withOpacity(0.2),
                borderRadius: BorderRadius.circular(4),
              ),
              child: const Text(
                '+50.5%',
                style: TextStyle(
                  color: PyrinColors.GREEN_COLOR,
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildActionButtons() {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 24),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          _buildActionButton('Send', 'assets/icons/send.svg'),
          _buildActionButton('Receive', 'assets/icons/receive.svg'),
          _buildActionButton('Swap', 'assets/icons/swap.svg'),
        ],
      ),
    );
  }

  Widget _buildActionButton(String label, String iconPath) {
    return Column(
      children: [
        Container(
          width: 48,
          height: 48,
          decoration: BoxDecoration(
            color: PyrinColors.BLACK1_COLOR,
            borderRadius: BorderRadius.circular(24),
          ),
          child: Center(
            child: SvgPicture.asset(
              iconPath,
              width: 24,
              height: 24,
              colorFilter: const ColorFilter.mode(PyrinColors.TEXT_COLOR, BlendMode.srcIn),
            ),
          ),
        ),
        const SizedBox(height: 8),
        Text(
          label,
          style: TextStyle(
            color: PyrinColors.TEXT_COLOR.withOpacity(0.7),
            fontSize: 12,
          ),
        ),
      ],
    );
  }

  Widget _buildHistoryTabs() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        color: PyrinColors.BLACK1_COLOR,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          Expanded(
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 12),
              decoration: BoxDecoration(
                color: PyrinColors.TEXT_COLOR,
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Text(
                'History',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: PyrinColors.BACKGROUND_COLOR,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
          Expanded(
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 12),
              child: Text(
                'Info',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: PyrinColors.TEXT_COLOR.withOpacity(0.7),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHistoryList() {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        _buildHistoryItem('Receive', '500 PYI', '\$49.98', '07 Oct 2021 at 02:32 AM', 'From: pyrin6f...a98'),
        _buildHistoryItem('Receive', '500 PYI', '\$50.17', '07 Oct 2021 at 02:32 AM', 'From: pyrin6f...a98'),
      ],
    );
  }

  Widget _buildHistoryItem(String type, String amount, String value, String date, String from) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      child: Row(
        children: [
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              color: PyrinColors.GREEN_COLOR.withOpacity(0.2),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Center(
              child: Icon(
                Icons.arrow_downward,
                color: PyrinColors.GREEN_COLOR,
                size: 20,
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  type,
                  style: const TextStyle(
                    color: PyrinColors.TEXT_COLOR,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  from,
                  style: TextStyle(
                    color: PyrinColors.TEXT_COLOR.withOpacity(0.7),
                    fontSize: 12,
                  ),
                ),
                Text(
                  date,
                  style: TextStyle(
                    color: PyrinColors.TEXT_COLOR.withOpacity(0.5),
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                amount,
                style: const TextStyle(
                  color: PyrinColors.TEXT_COLOR,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Text(
                value,
                style: TextStyle(
                  color: PyrinColors.TEXT_COLOR.withOpacity(0.7),
                  fontSize: 12,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildGoToMarketButton() {
    return Container(
      padding: const EdgeInsets.all(16),
      child: PyrinElevatedButton(
        text: "Go to home",
        onClick: () {
          // Handle go to market action
        },
        wide: true,
      ),
    );
  }
}
