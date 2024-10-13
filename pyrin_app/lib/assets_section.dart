import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:intl/intl.dart';
import 'package:pyrin_app/components/tokens_list.dart';
import 'package:pyrin_app/ui.dart';
import 'package:pyrin_app/core/token.dart';
import 'package:pyrin_app/core/section.dart';
import 'package:pyrin_app/assets_history.dart';

class AssetsSection extends StatelessWidget {
  const AssetsSection({super.key});

  @override
  Widget build(BuildContext context) {
    return SectionContainer(
      name: "Assets",
      child: Container(
        color: PyrinColors.BACKGROUND_COLOR,
        child: ListView(
          padding: const EdgeInsets.all(20),
          shrinkWrap: true,
          children: [
            const SizedBox(height: 20),
            _buildSearchBar(),
            const SizedBox(height: 20),
            _buildAssetsList(context),
          ],
        ),
      ),
    );
  }

  Widget _buildSearchBar() {
    return  PyrinTextField(
      hintText: "Search Token",
    );
  }

  Widget _buildAssetsList(BuildContext context) {
    final List<Token> tokens = [
      Token("Pyrin", "PYI", 150000000000),
      // Token("PyrinCoin", "PYC", (generateRandomBalance() * 1e8).round()),
      Token("PyrinCoin", "PYC", 0),
      // Token("PyrinX", "PYX", (generateRandomBalance() * 1e8).round()),
      Token("PyrinX", "PYX", 0),
    ];

    return PyrinListView<Token>(
      items: tokens,
      onItemClick: (token) {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => AssetsHistory(token: token),
          ),
        );
      },
      itemBuilder: (context, token) {
        return PyrinListViewItemBuilder(
          title: token.symbol,
          subtitle: token.name,
          icon: PyrinCircleIcon(
            darker: true,
            child: TokenIcon(symbol: token.symbol),
          ),
          leading: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (token.symbol == "PYI")
              SvgPicture.asset(
                "assets/icons/char.svg",
                width: 60,
                height: 30,
                colorFilter: const ColorFilter.mode(PyrinColors.GREEN_COLOR, BlendMode.srcIn),
              ),
              const SizedBox(width: 12),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    NumberFormat("#,##0.00", "en_US").format(token.amount / 1e8),
                    style: const TextStyle(color: PyrinColors.TEXT_COLOR, fontWeight: FontWeight.bold),
                  ),
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        '\$${NumberFormat("#,##0.00", "en_US").format(token.amount / 1e8 * 0.0149)}',
                        style: const TextStyle(color: PyrinColors.LIGHT_TEXT_COLOR, fontSize: 12),
                      ),
                      const SizedBox(width: 4),
        token.symbol == "PYI" ?
                        const Text(
                          '(+5.01%)',
                          style: TextStyle(color: PyrinColors.GREEN_COLOR, fontSize: 12),
                        ) : const Text(
                      '(+0.00%)',
                      style: TextStyle(color: PyrinColors.GREEN_COLOR, fontSize: 12),
                      ),
                    ],
                  ),
                ],
              ),
            ],
          ),
        );
      },
    );
  }
}

double generateRandomBalance() {
  return 0.01 + (1000.0 - 0.01) * math.Random().nextDouble();
}

Gradient getRandomGradient() {
  return _gradients[math.Random().nextInt(_gradients.length)];
}

const List<Gradient> _gradients = [
  LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [Color(0xFFB01EFF), Color(0xFFE1467C)],
  ),
  LinearGradient(
    begin: Alignment.topRight,
    end: Alignment.bottomLeft,
    colors: [Color(0xFFD079EE), Color(0xFF8A88FB)],
  ),
  LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [Color(0xFFC99FFF), Color(0xFF981ED2)],
  ),
  LinearGradient(
    begin: Alignment.bottomCenter,
    end: Alignment.topCenter,
    colors: [Color(0xFF9B23EA), Color(0xFF5F72BD)],
  ),
  LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [Color(0xFFB39FFF), Color(0xFF6A1ED2)],
  ),
  LinearGradient(
    begin: Alignment.topRight,
    end: Alignment.bottomLeft,
    colors: [Color(0xFF4300B1), Color(0xFFA531DC)],
  ),
  LinearGradient(
    begin: Alignment.bottomLeft,
    end: Alignment.topRight,
    colors: [Color(0xFF764BA2), Color(0xFF667EEA)],
  ),
];
