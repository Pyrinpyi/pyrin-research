import "package:flutter/material.dart";
import "package:flutter_svg/flutter_svg.dart";
import "package:provider/provider.dart";
import "package:pyrin_app/core/addressbook.dart";
import "package:pyrin_app/core/section.dart";
import "package:pyrin_app/core/wallet_provider.dart";
import "package:pyrin_app/ui.dart";

class MenuItem extends StatelessWidget
{
    final String icon;
    final String name;
    final String path;

    MenuItem({required this.icon, required this.name, required this.path});

    @override
    Widget build(BuildContext context)
    {
        return InkWell(
          onTap: () => {},
          child: Container(
              padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 20),
              margin: const EdgeInsets.symmetric(vertical: 5),
              child: Row(
                children: [
                    SvgPicture.asset("assets/icons/$icon.svg"),
                    const SizedBox(width: 20),
                    Text(name, style: Theme.of(context).textTheme.bodyMedium!.copyWith(fontWeight: FontWeight.w700)),
                    Expanded(child: Container()),
                    SvgPicture.asset("assets/icons/arrow-small-right.svg"),
                ],
              ),
          ),
        );
    }
}

class MenuSection extends StatelessWidget
{
    @override
    Widget build(BuildContext context)
    {
        return SectionContainer(
            name: "Menu",
            child: Container(
              child: ListView(
                shrinkWrap: true,
                children: [
                  Container(
                    padding: EdgeInsets.symmetric(horizontal: 20),
                    child: Consumer<WalletProvider>(
                      builder: (context, wallet, child)
                      {
                        return PyrinListViewItem(
                          title: "Wallet",
                          subtitle: AddressBook.shortenAddress(wallet.receiveAddress),
                          icon: Avatar(size: 60),
                          // leading: PyrinElevatedButton(
                          //   width: 100,
                          //   text: "Switch",
                          //   onClick: () {},
                          // ),
                        );
                      },
                    ),
                  ),
                  const SizedBox(height: 20),
                  // Row(
                  //   children: [
                  //     Avatar(),
                  //     const SizedBox(width: 20),
                  //     Column(
                  //       crossAxisAlignment: CrossAxisAlignment.start,
                  //       children: [
                  //         Text("Satoshi Nakamoto", style: Theme.of(context).textTheme.bodyMedium!.copyWith(fontWeight: FontWeight.w600)),
                  //         Text("pyrintest:qqgjen34j4uqvyece06675mzxkc6fnaxrfmaptl6lcsmv6un7uc6vvwnsscs2", style: Theme.of(context).textTheme.bodyMedium),
                  //       ],
                  //     )
                  //   ],
                  // )
                  MenuItem(icon: "task", name: "Activity Log", path: "/activity-log"),
                  MenuItem(icon: "setting", name: "General", path: "/general"),
                  MenuItem(icon: "sun", name: "Preferences", path: "/preferences"),
                  MenuItem(icon: "key", name: "Security & Privacy", path: "/security-privacy"),
                  MenuItem(icon: "notification", name: "Push Notifications", path: "/push-notifications"),
                  MenuItem(icon: "lovely", name: "Donation", path: "/donation"),
                  MenuItem(icon: "life-buoy", name: "Help", path: "/help"), // TODO: support@pyrin.network

                  Container(
                    width: double.infinity,
                    height: 1,
                    color: PyrinColors.WHITE_COLOR.withOpacity(0.06),
                    margin: const EdgeInsets.only(left: 20, right: 20, bottom: 20),
                  ),
                  MenuItem(icon: "logout", name: "Logout", path: "/logout"),
                  Container(
                    margin: const EdgeInsets.only(top: 40, bottom: 10),
                    padding: const EdgeInsets.all(20),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                          SvgPicture.asset(
                              "assets/pyrin.svg",
                            width: 100,
                            colorFilter: ColorFilter.mode(PyrinColors.WHITE_COLOR, BlendMode.srcIn),
                          ),
                          Text(
                            "All rights reserved © 2024",
                            style: Theme.of(context).textTheme.bodyMedium!.copyWith(
                                color: PyrinColors.TEXT_COLOR.withOpacity(0.2),
                                fontSize: 11
                            ),
                          )
                      ],
                    ),
                  )
                ],
              ),
            )
        );
    }
}
